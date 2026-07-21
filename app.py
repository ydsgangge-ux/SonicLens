"""
音频分析后端 — librosa 专业特征提取 + DeepSeek / OpenAI AI 分析
Windows / macOS / Linux 全平台兼容
运行: python app.py   访问: http://localhost:8000
"""
import os, re, json, tempfile, traceback
from pathlib import Path

import numpy as np
import librosa
import librosa.feature
import librosa.effects
import librosa.onset

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI(title="librosa 音频分析器")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")


# ─────────────────────────────────────────────────────────
# librosa 特征提取
# ─────────────────────────────────────────────────────────

def extract_features(path: str) -> dict:
    """用 librosa 提取专业音频特征，对标 Essentia 核心功能"""

    # 加载音频（单声道，22050Hz，最多分析前90秒）
    y, sr = librosa.load(path, sr=22050, mono=True, duration=90)
    duration = librosa.get_duration(y=y, sr=sr)

    features = {"duration": round(duration, 2), "sample_rate": sr}

    # ── 1. BPM & 节拍 ────────────────────────────────────
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    # 兼容 numpy 不同版本，tempo 可能是数组或标量
    if isinstance(tempo, np.ndarray):
        features["bpm"] = round(float(tempo[0]), 2)
    else:
        features["bpm"] = round(float(tempo), 2)
    features["beats_count"] = len(beats)

    # 节奏规律性（节拍间隔的稳定程度）
    if len(beats) > 2:
        intervals = np.diff(beats)
        regularity = float(1.0 - np.std(intervals) / (np.mean(intervals) + 1e-8))
        features["rhythm_regularity"] = round(max(0.0, min(1.0, regularity)), 3)
    else:
        features["rhythm_regularity"] = 0.0

    # Onset 强度（节拍冲击感）
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    features["onset_strength_mean"] = round(float(np.mean(onset_env)), 4)
    features["onset_strength_std"]  = round(float(np.std(onset_env)), 4)

    # ── 2. 调性 & 和声（Krumhansl-Schmuckler）────────────
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36)
    chroma_mean = chroma.mean(axis=1)
    chroma_norm = chroma_mean / (chroma_mean.max() + 1e-8)

    NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    MAJ = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    MIN = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

    def ks_corr(profile, chroma_vec):
        scores = []
        for i in range(12):
            shifted = np.roll(chroma_vec, -i)
            c = np.corrcoef(shifted, profile)[0, 1]
            scores.append(float(c) if not np.isnan(c) else 0.0)
        return scores

    maj_scores = ks_corr(MAJ, chroma_norm)
    min_scores = ks_corr(MIN, chroma_norm)
    best_maj_idx = int(np.argmax(maj_scores))
    best_min_idx = int(np.argmax(min_scores))

    if maj_scores[best_maj_idx] >= min_scores[best_min_idx]:
        key_idx  = best_maj_idx
        is_major = True
        key_str  = f"{NOTE_NAMES[key_idx]} 大调"
        key_strength = round(maj_scores[best_maj_idx], 3)
    else:
        key_idx  = best_min_idx
        is_major = False
        key_str  = f"{NOTE_NAMES[key_idx]} 小调"
        key_strength = round(min_scores[best_min_idx], 3)

    features["key"]        = NOTE_NAMES[key_idx]
    features["scale"]      = "major" if is_major else "minor"
    features["key_full"]   = key_str
    features["key_strength"] = key_strength
    features["key_idx"]    = key_idx
    features["chroma"]     = [round(float(v), 4) for v in chroma_norm]

    # ── 3. MFCC（40维音色指纹）──────────────────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_delta  = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    features["mfcc_mean"]   = [round(float(v), 3) for v in mfcc.mean(axis=1)]
    features["mfcc_std"]    = [round(float(v), 3) for v in mfcc.std(axis=1)]
    features["mfcc_delta"]  = [round(float(v), 3) for v in mfcc_delta.mean(axis=1)]
    features["mfcc_delta2"] = [round(float(v), 3) for v in mfcc_delta2.mean(axis=1)]

    # ── 4. 频谱特征（逐帧统计）──────────────────────────
    stft  = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    # 谱质心
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=512)[0]
    features["spectral_centroid_mean"] = round(float(centroid.mean()), 1)
    features["spectral_centroid_std"]  = round(float(centroid.std()), 1)

    # 谱滚降 85%
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    features["spectral_rolloff_mean"] = round(float(rolloff.mean()), 1)

    # 谱平坦度
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    features["spectral_flatness_mean"] = round(float(flatness.mean()), 6)

    # 谱带宽
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features["spectral_bandwidth_mean"] = round(float(bandwidth.mean()), 1)

    # 谱对比度（7个子带）
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    features["spectral_contrast"] = [round(float(v), 3) for v in contrast.mean(axis=1)]

    # 谱流量（音色变化速率）
    flux = np.mean(np.sqrt(np.sum(np.diff(stft, axis=1)**2, axis=0)))
    features["spectral_flux_mean"] = round(float(flux), 4)

    # ── 5. 8段精细频带能量 ──────────────────────────────
    band_defs = [
        ("sub_bass_20_60",    20,    60),
        ("bass_60_250",       60,   250),
        ("low_mid_250_500",  250,   500),
        ("mid_500_2k",       500,  2000),
        ("upper_mid_2k_4k", 2000,  4000),
        ("presence_4k_8k",  4000,  8000),
        ("brilliance_8k_16k",8000,16000),
        ("air_16k_20k",    16000, 20000),
    ]
    avg_spec = stft.mean(axis=1)
    raw_bands = {}
    for name, lo, hi in band_defs:
        mask = (freqs >= lo) & (freqs < hi)
        raw_bands[name] = float(avg_spec[mask].mean()) if mask.any() else 0.0
    max_e = max(raw_bands.values()) or 1.0
    features["freq_bands_8"] = {k: round(v / max_e * 100, 1) for k, v in raw_bands.items()}

    hi_e  = sum(v for k, v in features["freq_bands_8"].items() if any(x in k for x in ["4k","8k","16k","air"]))
    lo_e  = sum(v for k, v in features["freq_bands_8"].items() if any(x in k for x in ["sub","bass_60"]))
    tot_e = sum(features["freq_bands_8"].values()) or 1
    features["brightness_ratio"] = round(hi_e / tot_e, 3)
    features["bass_ratio"]       = round(lo_e / tot_e, 3)

    # ── 6. 响度 & 动态 ──────────────────────────────────
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    features["rms_mean"] = round(float(rms.mean()), 5)
    features["rms_std"]  = round(float(rms.std()), 5)
    features["loudness_db"] = round(float(20 * np.log10(rms.mean() + 1e-8)), 2)

    p90 = np.percentile(rms, 90)
    p10 = np.percentile(rms, 10)
    features["dynamic_range_db"] = round(float(20 * np.log10((p90 + 1e-8) / (p10 + 1e-8))), 2)

    # ── 7. 响度动态轮廓（时间序列，10段）─────────────────
    frames_total = len(rms)
    segments = 10
    frame_per_segment = max(frames_total // segments, 1)
    contour = []
    for i in range(segments):
        start = i * frame_per_segment
        end = min(start + frame_per_segment, frames_total)
        if end > start:
            contour.append(round(float(np.mean(rms[start:end])), 4))
        else:
            contour.append(round(float(rms[-1]), 4))
    features["dynamic_contour"] = contour

    # ── 8. 音高 & 旋律性（pyin 算法）───────────────────
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr, hop_length=512
        )
        voiced_f0 = f0[voiced_flag & (f0 > 0)]
        if len(voiced_f0) > 5:
            features["pitch_mean_hz"]    = round(float(np.mean(voiced_f0)), 1)
            features["pitch_std_hz"]     = round(float(np.std(voiced_f0)), 1)
            features["pitch_range_hz"]   = round(float(np.max(voiced_f0) - np.min(voiced_f0)), 1)
            features["melodic_strength"] = round(float(voiced_flag.mean()), 3)
        else:
            features["pitch_mean_hz"] = 0
            features["pitch_std_hz"]  = 0
            features["pitch_range_hz"]= 0
            features["melodic_strength"] = 0.0
    except Exception:
        features["pitch_mean_hz"] = 0
        features["pitch_std_hz"]  = 0
        features["pitch_range_hz"]= 0
        features["melodic_strength"] = 0.0

    # ── 9. 零交叉率 ─────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=512)[0]
    features["zcr_mean"] = round(float(zcr.mean()), 5)
    features["zcr_std"]  = round(float(zcr.std()), 5)

    # ── 10. 泛音 & 打击乐分离（能量比）──────────────────
    y_harm, y_perc = librosa.effects.hpss(y)
    harm_energy = float(np.mean(y_harm ** 2))
    perc_energy = float(np.mean(y_perc ** 2))
    total_hp    = harm_energy + perc_energy + 1e-8
    features["harmonic_ratio"]   = round(harm_energy / total_hp, 3)
    features["percussive_ratio"] = round(perc_energy / total_hp, 3)

    # ── 11. 可舞性估算 ───────────────────────────────────
    bpm_dance  = 1.0 - abs(features["bpm"] - 120) / 120.0
    bpm_dance  = max(0.0, min(1.0, bpm_dance))
    danceability = (
        features["rhythm_regularity"] * 0.4 +
        bpm_dance * 0.3 +
        features["percussive_ratio"] * 0.3
    ) * 3.0
    features["danceability"] = round(danceability, 3)

    # ── 12. 和声不和谐度估算 ─────────────────────────────
    dissonance = float(flatness.mean()) * 0.5 + (1.0 - key_strength) * 0.5
    features["dissonance"] = round(min(1.0, dissonance), 4)

    # ── 13. 谱复杂度 ────────────────────────────────────
    complexity = float(np.mean(np.sum(stft > stft.mean() * 0.1, axis=0)))
    features["spectral_complexity_mean"] = round(complexity / (stft.shape[0] + 1e-8) * 100, 2)

    return features


# ─────────────────────────────────────────────────────────
# AI 深度分析（DeepSeek / OpenAI 兼容接口）
# ─────────────────────────────────────────────────────────

DEEPSEEK_MODELS = {"deepseek-chat", "deepseek-reasoner"}


def _repair_json(raw: str) -> str:
    """尝试修复 AI 返回的常见 JSON 格式错误"""
    fixed = raw
    # 1. 修复 "key": "value\n第二行" → "key": "value 第二行" (字符串内未转义换行)
    #    在 JSON 值字符串内部把裸换行替换为空格
    in_str = False
    i = 0
    out = []
    while i < len(fixed):
        ch = fixed[i]
        if ch == '\\' and in_str and i + 1 < len(fixed) and fixed[i+1] == '"':
            out.append(ch); out.append(fixed[i+1]); i += 2; continue
        if ch == '"' and (i == 0 or fixed[i-1] != '\\'):
            in_str = not in_str
        if in_str and ch in ('\n', '\r'):
            out.append(' ')
        else:
            out.append(ch)
        i += 1
    fixed = ''.join(out)
    # 2. 删除尾逗号  },  →  }
    fixed = re.sub(r',\s*}', '}', fixed)
    fixed = re.sub(r',\s*]', ']', fixed)
    # 3. 修复缺少逗号:  "value" "key"  →  "value", "key"
    fixed = re.sub(r'"\s*\n\s*"', '",\n"', fixed)
    return fixed


def _extract_json_fields(raw: str, exc: json.JSONDecodeError) -> dict:
    """最后兜底：用正则提取 "key": "value..." 字段"""
    result = {}
    # 匹配 "key": "value..." 或 "key": 数值
    for m in re.finditer(r'"(\w+)":\s*(?:"((?:[^"\\]|\\.)*)"|([\d.\-]+))', raw):
        key = m.group(1)
        val = m.group(2) if m.group(2) is not None else m.group(3)
        try:
            result[key] = float(val) if '.' in val else int(val)
        except (ValueError, TypeError):
            result[key] = val
    # 如果正则也没提取到任何字段，把原始错误抛出
    if not result:
        raise exc
    return result



def analyze_with_ai(features, filename, api_key, model):
    # 初始化 OpenAI 客户端
    if model in DEEPSEEK_MODELS:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
    else:
        client = OpenAI(api_key=api_key)

    f = features
    prompt = f"""你是世界顶级的音乐评论家、乐理学家和音频工程师，兼具学术深度与文学表达能力。
你的分析风格参考：《Time》by Hans Zimmer 的乐评——有结构层次、有技法剖析、有情感叙事、有文化视野。

以下是用 librosa 从音频文件 "{filename}" 中提取的精确声学特征数据：

━━━ 声学特征数据 ━━━

节奏：BPM {f['bpm']} | 节拍数 {f['beats_count']} | 规律性 {f['rhythm_regularity']} | Onset强度 {f['onset_strength_mean']}±{f['onset_strength_std']} | 可舞性 {f['danceability']} | 时长 {f['duration']}s
调性：{f['key_full']} | K-S强度 {f['key_strength']} | 不和谐度 {f['dissonance']} | 泛音/打击乐比 {f['harmonic_ratio']}/{f['percussive_ratio']}
频谱：质心 {f['spectral_centroid_mean']}±{f['spectral_centroid_std']}Hz | 滚降 {f['spectral_rolloff_mean']}Hz | 平坦度 {f['spectral_flatness_mean']} | 带宽 {f['spectral_bandwidth_mean']}Hz | 流量 {f['spectral_flux_mean']} | 复杂度 {f['spectral_complexity_mean']}
频带(%)：次低频{f['freq_bands_8'].get('sub_bass_20_60',0)} | 低频{f['freq_bands_8'].get('bass_60_250',0)} | 中低{f['freq_bands_8'].get('low_mid_250_500',0)} | 中频{f['freq_bands_8'].get('mid_500_2k',0)} | 中高{f['freq_bands_8'].get('upper_mid_2k_4k',0)} | 存在感{f['freq_bands_8'].get('presence_4k_8k',0)} | 亮度{f['freq_bands_8'].get('brilliance_8k_16k',0)} | 空气{f['freq_bands_8'].get('air_16k_20k',0)}
明亮度比 {f['brightness_ratio']} | 低频比 {f['bass_ratio']}
MFCC均值(前13维)：{f.get('mfcc_mean', [])[:13]}
MFCC标准差：{f.get('mfcc_std', [])[:13]}
MFCC Delta：{f.get('mfcc_delta', [])[:13]}
动态：RMS {f['rms_mean']}±{f['rms_std']} | 响度 {f['loudness_db']}dB | 动态范围 {f['dynamic_range_db']}dB
响度动态轮廓（10段时序能量）：{f.get('dynamic_contour', [])}
音高：均值 {f.get('pitch_mean_hz',0)}Hz | 标准差 {f.get('pitch_std_hz',0)}Hz | 范围 {f.get('pitch_range_hz',0)}Hz | 旋律性 {f.get('melodic_strength',0)}
ZCR：{f['zcr_mean']}±{f['zcr_std']}
谱对比度(7段)：{f.get('spectral_contrast', [])}

━━━ 输出要求 ━━━

请输出严格的 JSON，所有文字字段都要写得像一篇专业乐评，有深度、有细节、有文学性。
每个分析字段的要求：
- 必须引用上面的具体数值作为论据
- 用专业术语但同时保持可读性
- 像分析《Time》那样：先描述技法，再讲情感内核，再说它为什么这样设计
- 每个分析字段不少于 3 句话，重要字段（结构分析、情感内核）不少于 5 句

输出 JSON（不含任何其他文字）：
{{
  "summary": "5-6句总体评述：先用一句话定性这首音乐的气质，再分别从节奏、调性、音色、动态四个维度各用一句话点出最突出的特征，最后一句给出整体评价定论",

  "structure_analysis": "【音乐结构分析】深度剖析这首音乐的结构逻辑：基于动态范围{f['dynamic_range_db']}dB、RMS标准差{f['rms_std']}、Onset强度标准差{f['onset_strength_std']}判断是否有明显的段落起伏（引子/发展/高潮/消退）。响度动态轮廓{f.get('dynamic_contour',[])}揭示了能量如何随时间推移而变化——是持续攀升、波浪起伏、还是渐弱消散？描述音乐是如何随时间展开的，能量如何累积或释放，旋律如何推进或循环。如果是循环递进结构，要分析其叠加逻辑",

  "key_harmony_analysis": "【调性与和声分析】从{f['key_full']}（调性强度{f['key_strength']}）出发，分析该调式的色彩特质与情绪倾向。结合不和谐度{f['dissonance']}和谱复杂度{f['spectral_complexity_mean']}分析和声密度——是简洁的三和弦还是复杂的扩展和弦？谱对比度{f.get('spectral_contrast',[])}反映了哪些和声层次？推断可能的和声风格（古典、爵士、流行、现代）",

  "rhythm_groove_analysis": "【节奏与律动分析】BPM {f['bpm']} 对应什么样的人体感受（心跳、呼吸、步行节奏）？节奏规律性{f['rhythm_regularity']}说明它是机械精准还是人性摇摆？泛音/打击乐比{f['harmonic_ratio']}/{f['percussive_ratio']}揭示了节奏的驱动方式——是旋律主导还是鼓点主导？Onset强度{f['onset_strength_mean']}反映了节拍的冲击感",

  "timbre_texture_analysis": "【音色与织体分析】基于MFCC均值{f.get('mfcc_mean',[][:3])}判断主要音色特质（明亮/温暖/暗沉/金属）；MFCC标准差{f.get('mfcc_std',[][:3])}说明音色在时间轴上是稳定还是多变；谱平坦度{f['spectral_flatness_mean']}说明是纯音乐器还是噪声成分；谱质心标准差{f['spectral_centroid_std']}Hz说明音色层次的丰富程度；泛音比{f['harmonic_ratio']}说明音色的纯净度",

  "dynamics_emotion_analysis": "【动态与情感张力】动态范围{f['dynamic_range_db']}dB意味着什么？（<6dB=电台压缩，6-12dB=流行标准，12-18dB=古典自然，>18dB=极致动态）RMS标准差{f['rms_std']}反映了情绪波动的幅度。响度动态轮廓{f.get('dynamic_contour',[])}展现了能量如何随时间流动——是缓慢蓄力后爆发，还是持续平稳？结合频带分布，分析这首音乐是如何通过动态变化制造情感张力的",

  "melody_narrative_analysis": "【旋律与叙事性】旋律性强度{f.get('melodic_strength',0)}说明这首音乐有多少时间处于可辨识的音高（越高越'有旋律'）。音高范围{f.get('pitch_range_hz',0)}Hz对应几个八度的跨度？音高标准差{f.get('pitch_std_hz',0)}Hz说明旋律线条是平稳抒情还是起伏剧烈？这些数据共同描绘出什么样的旋律叙事风格",

  "freq_landscape": "【频谱景观描述】把8段频带数据转化为一幅声音的地图：哪个频段是这首音乐的'地基'（最厚重），哪个是'骨架'（最具存在感），哪个是'空气'（最通透）。明亮度比{f['brightness_ratio']}说明这是温暖厚重的声音还是清亮开阔的声音。指出任何明显缺失的频段，这种缺失是风格选择还是制作问题",

  "tempo": {f['bpm']},
  "tempoFeel": "意大利速度术语 + 中文名称 + 一句话描述这个速度给人的具体身体感受（如：像慢走的步伐，像呼吸的节律）",
  "key": "{f['key_full']}",
  "mode_desc": "3句话：第1句说这个调式的理论特征（音阶构成、特征音级），第2句说它在音乐历史中的情感传统，第3句结合本曲的具体数据说它在这里呈现出什么独特效果",

  "chords": ["根据Chroma向量{f.get('chroma',[])}和调性精确推断6-8个和弦，格式：'Am（i级，主和弦）'，要包含罗马数字功能标注"],
  "chord_progression_desc": "2-3句话：第1句描述和弦进行的模式（是否有常见的I-V-vi-IV？是否有借用和弦？），第2句分析这个进行制造的情感运动方向（解决感/悬浮感/循环感），第3句说这种和声设计的美学意图",

  "emotions": [
    {{"name": "情感名（2-4个字）", "score": 0到100的整数, "desc": "3句话分析：第1句说数据依据，第2句说这种情感如何体现在音乐中，第3句说为什么听者会有这种感受", "color": "blue或amber或teal或coral或purple或pink"}},
    {{"name": "情感名", "score": 整数, "desc": "同上格式", "color": "颜色"}},
    {{"name": "情感名", "score": 整数, "desc": "同上格式", "color": "颜色"}},
    {{"name": "情感名", "score": 整数, "desc": "同上格式", "color": "颜色"}},
    {{"name": "情感名", "score": 整数, "desc": "同上格式", "color": "颜色"}},
    {{"name": "情感名", "score": 整数, "desc": "同上格式", "color": "颜色"}}
  ],

  "keywords": ["10-12个关键词，覆盖：情感形容词、音乐风格、乐器特征、场景描述、时代感"],

  "instruments": [
    {{"name": "乐器名", "confidence": 整数, "reason": "3句话：第1句说MFCC/ZCR/谱特征的具体数值依据，第2句说这个乐器在这首音乐中扮演什么角色，第3句说判断的不确定性"}},
    {{"name": "乐器名", "confidence": 整数, "reason": "同上格式"}},
    {{"name": "乐器名", "confidence": 整数, "reason": "同上格式"}},
    {{"name": "乐器名", "confidence": 整数, "reason": "同上格式"}}
  ],

  "music_style": "精确风格/流派，格式：'主流派 · 子风格'（如 'Neo-Classical · 极简主义交响'）",

  "production_notes": "给音乐制作人的4句专业建议：第1句基于频带数据的混音问题，第2句基于动态范围的母带建议，第3句基于音色特征的编曲建议，第4句基于整体风格的参考制作方向",

  "similar_artists": ["5个风格相近的艺术家/作品，格式：'艺术家名 — 代表作'"]
}}"""

    resp = client.chat.completions.create(
        model=model,
        max_tokens=6000,
        temperature=0.4,
        messages=[
            {"role": "system", "content": "你是兼具学术深度与文学表达的顶级音乐评论家。你的分析风格：技法严谨、情感细腻、文字有质感。每个分析字段都要写出深度，不能流于表面，要像一篇发表在专业音乐杂志上的乐评。只输出纯JSON，不含任何额外文字、注释或markdown代码块。"},
            {"role": "user",   "content": prompt}
        ]
    )
    raw = resp.choices[0].message.content or ""
    raw = raw.replace("```json", "").replace("```", "").strip()
    s, e = raw.find("{"), raw.rfind("}")
    if s != -1 and e != -1:
        raw = raw[s:e+1]
    # 清洗 AI 返回文本中的非法控制字符（保留 \n \r \t）
    raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # ── Fallback: 尝试修复常见 JSON 错误 ──
    fixed = _repair_json(raw)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError as exc:
        # 最后兜底：用正则提取所有 "key": "value" 组装
        return _extract_json_fields(raw, exc)


# ─────────────────────────────────────────────────────────
# API 路由
# ─────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("static/index.html")

# 支持的音频扩展名（快速白名单，命中即直接用）
AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".opus",
              ".aiff", ".aif", ".wma", ".wv", ".ape", ".caf", ".webm", ".mp4", ".amr"}

def _sniff_audio_ext(data: bytes):
    """通过文件头魔数识别真实音频格式，兼容无扩展名 / 扩展名错误的情况"""
    if len(data) < 12:
        return None
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return ".wav"
    if data[:4] == b"FORM" and data[8:12] in (b"AIFF", b"AIFC"):
        return ".aiff"
    if data[:4] == b"fLaC":
        return ".flac"
    if data[:4] == b"OggS":
        return ".ogg"
    if data[:4] == b"ftyp":
        return ".m4a"
    if data[:3] == b"ID3" or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return ".mp3"
    return None


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    api_key: str = Form(default=""),
    model: str = Form(default="deepseek-chat")
):
    raw = await file.read()
    filename = file.filename or "audio"
    suffix = Path(filename).suffix.lower()
    ext = suffix if suffix in AUDIO_EXTS else _sniff_audio_ext(raw)
    if ext is None:
        raise HTTPException(
            400,
            "无法识别的音频格式，请上传 MP3 / WAV / FLAC / OGG / M4A / AAC / OPUS 等常见音频文件",
        )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        features   = extract_features(tmp_path)
        ai_result  = None
        if api_key.strip():
            ai_result = analyze_with_ai(features, filename, api_key.strip(), model)
        return {"success": True, "filename": filename, "features": features, "analysis": ai_result}
    except Exception as e:
        raise HTTPException(500, f"分析失败: {str(e)}\n{traceback.format_exc()}")
    finally:
        os.unlink(tmp_path)

@app.get("/api/health")
async def health():
    return {"status": "ok", "backend": "librosa", "librosa_version": librosa.__version__}

if __name__ == "__main__":
    import uvicorn
    # reload=False：在该环境下 reload 模式的 worker 子进程无法正常就绪，
    # 会导致服务进程在跑、端口已绑定，但浏览器始终连不上。
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
