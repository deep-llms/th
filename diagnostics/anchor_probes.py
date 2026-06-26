"""EmbHub Anchor Probes — Global Usage & Cross-Lingual Overlap.

Probe 1: Global & per-language anchor usage on full eval sets (10M tokens/lang)
Probe 2: Cross-lingual anchor overlap on translation pairs

Usage:
  python diagnostics/anchor_probes.py \
      --checkpoints /path/to/S3/checkpoint-1500 /path/to/S3/checkpoint-3400 /path/to/S3/checkpoint-5500 \
      --eval-dir data/Qwen_Qwen3-0.6B/eval
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from scipy import stats
from datasets import load_from_disk
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from model_wrapper_v2 import inject_embhub, EMBHUB_WEIGHTS_NAME, EMBHUB_CONFIG_NAME

LANGS = ["en", "vi", "zh", "ru", "de", "ar"]

# Translation tuples: (en, vi, zh, ru, de, ar)
# Curated for common concrete nouns likely to be single-token
TRANSLATION_TUPLES = [
    # Nature
    ("water", "nước", "水", "вода", "Wasser", "ماء"),
    ("fire", "lửa", "火", "огонь", "Feuer", "نار"),
    ("sun", "mặt trời", "太阳", "солнце", "Sonne", "شمس"),
    ("moon", "mặt trăng", "月亮", "луна", "Mond", "قمر"),
    ("tree", "cây", "树", "дерево", "Baum", "شجرة"),
    ("rain", "mưa", "雨", "дождь", "Regen", "مطر"),
    ("wind", "gió", "风", "ветер", "Wind", "رياح"),
    ("earth", "đất", "地球", "земля", "Erde", "أرض"),
    ("snow", "tuyết", "雪", "снег", "Schnee", "ثلج"),
    ("river", "sông", "河", "река", "Fluss", "نهر"),
    ("star", "sao", "星", "звезда", "Stern", "نجم"),
    ("sky", "trời", "天", "небо", "Himmel", "سماء"),
    ("sea", "biển", "海", "море", "Meer", "بحر"),
    ("mountain", "núi", "山", "гора", "Berg", "جبل"),
    ("flower", "hoa", "花", "цветок", "Blume", "زهرة"),
    ("forest", "rừng", "森林", "лес", "Wald", "غابة"),
    ("island", "đảo", "岛", "остров", "Insel", "جزيرة"),
    ("cloud", "mây", "云", "облако", "Wolke", "سحابة"),
    ("ice", "băng", "冰", "лёд", "Eis", "جليد"),
    ("sand", "cát", "沙", "песок", "Sand", "رمل"),
    # Animals
    ("fish", "cá", "鱼", "рыба", "Fisch", "سمكة"),
    ("bird", "chim", "鸟", "птица", "Vogel", "طائر"),
    ("dog", "chó", "狗", "собака", "Hund", "كلب"),
    ("cat", "mèo", "猫", "кошка", "Katze", "قطة"),
    ("horse", "ngựa", "马", "лошадь", "Pferd", "حصان"),
    ("snake", "rắn", "蛇", "змея", "Schlange", "ثعبان"),
    ("cow", "bò", "牛", "корова", "Kuh", "بقرة"),
    ("sheep", "cừu", "羊", "овца", "Schaf", "خروف"),
    ("lion", "sư tử", "狮子", "лев", "Löwe", "أسد"),
    ("bear", "gấu", "熊", "медведь", "Bär", "دب"),
    # Body
    ("hand", "tay", "手", "рука", "Hand", "يد"),
    ("eye", "mắt", "眼", "глаз", "Auge", "عين"),
    ("heart", "tim", "心", "сердце", "Herz", "قلب"),
    ("head", "đầu", "头", "голова", "Kopf", "رأس"),
    ("blood", "máu", "血", "кровь", "Blut", "دم"),
    ("mouth", "miệng", "嘴", "рот", "Mund", "فم"),
    ("ear", "tai", "耳", "ухо", "Ohr", "أذن"),
    ("nose", "mũi", "鼻", "нос", "Nase", "أنف"),
    ("tooth", "răng", "牙", "зуб", "Zahn", "سن"),
    ("hair", "tóc", "头发", "волосы", "Haar", "شعر"),
    # Objects
    ("house", "nhà", "房子", "дом", "Haus", "منزل"),
    ("book", "sách", "书", "книга", "Buch", "كتاب"),
    ("door", "cửa", "门", "дверь", "Tür", "باب"),
    ("stone", "đá", "石", "камень", "Stein", "حجر"),
    ("gold", "vàng", "金", "золото", "Gold", "ذهب"),
    ("milk", "sữa", "奶", "молоко", "Milch", "حليب"),
    ("bread", "bánh mì", "面包", "хлеб", "Brot", "خبز"),
    ("knife", "dao", "刀", "нож", "Messer", "سكين"),
    ("ship", "tàu", "船", "корабль", "Schiff", "سفينة"),
    ("bridge", "cầu", "桥", "мост", "Brücke", "جسر"),
    # People & concepts
    ("king", "vua", "王", "король", "König", "ملك"),
    ("food", "thức ăn", "食物", "еда", "Essen", "طعام"),
    ("road", "đường", "路", "дорога", "Straße", "طريق"),
    ("night", "đêm", "夜", "ночь", "Nacht", "ليل"),
    ("mother", "mẹ", "母亲", "мать", "Mutter", "أم"),
    ("father", "cha", "父亲", "отец", "Vater", "أب"),
    ("child", "trẻ em", "孩子", "ребёнок", "Kind", "طفل"),
    ("friend", "bạn", "朋友", "друг", "Freund", "صديق"),
    ("enemy", "kẻ thù", "敌人", "враг", "Feind", "عدو"),
    ("doctor", "bác sĩ", "医生", "врач", "Arzt", "طبيب"),
    ("teacher", "giáo viên", "老师", "учитель", "Lehrer", "معلم"),
    ("soldier", "lính", "士兵", "солдат", "Soldat", "جندي"),
    # Actions / states
    ("war", "chiến tranh", "战争", "война", "Krieg", "حرب"),
    ("peace", "hòa bình", "和平", "мир", "Frieden", "سلام"),
    ("love", "tình yêu", "爱", "любовь", "Liebe", "حب"),
    ("death", "cái chết", "死亡", "смерть", "Tod", "موت"),
    ("life", "cuộc sống", "生活", "жизнь", "Leben", "حياة"),
    ("time", "thời gian", "时间", "время", "Zeit", "وقت"),
    ("money", "tiền", "钱", "деньги", "Geld", "مال"),
    ("city", "thành phố", "城市", "город", "Stadt", "مدينة"),
    # Colors
    ("red", "đỏ", "红", "красный", "rot", "أحمر"),
    ("blue", "xanh", "蓝", "синий", "blau", "أزرق"),
    ("white", "trắng", "白", "белый", "weiß", "أبيض"),
    ("black", "đen", "黑", "чёрный", "schwarz", "أسود"),
    ("green", "xanh lá", "绿", "зелёный", "grün", "أخضر"),
    # Numbers
    ("one", "một", "一", "один", "eins", "واحد"),
    ("two", "hai", "二", "два", "zwei", "اثنان"),
    ("three", "ba", "三", "три", "drei", "ثلاثة"),
    ("ten", "mười", "十", "десять", "zehn", "عشرة"),
    ("hundred", "trăm", "百", "сто", "hundert", "مائة"),
]


def load_checkpoint(checkpoint_path, device="cpu"):
    config = AutoConfig.from_pretrained(checkpoint_path)
    embhub_cfg_path = os.path.join(checkpoint_path, EMBHUB_CONFIG_NAME)
    embhub_wt_path = os.path.join(checkpoint_path, EMBHUB_WEIGHTS_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, config=config, torch_dtype=torch.bfloat16
    )

    with open(embhub_cfg_path) as f:
        hub_cfg = json.load(f)
    hub = inject_embhub(model, num_embeddings=hub_cfg["num_embeddings"], alpha=hub_cfg["alpha"])
    hub.load_state_dict(torch.load(embhub_wt_path, map_location="cpu", weights_only=True))

    model.to(device)
    model.eval()
    return model, hub


def get_anchor_weights(hub, token_embeddings):
    """Reproduce the selection path in fp32."""
    with torch.no_grad():
        q = F.normalize(token_embeddings.float(), dim=-1)
        k = F.normalize(hub.hub_embeddings.float(), dim=-1)
        scale = hub.log_logit_scale.float().exp().clamp(max=100.0)
        logits = (q @ k.T) * scale
        weights = logits.softmax(dim=-1)
    return weights


# -----------------------------------------------------------------------
# PROBE 1 — Global & per-language anchor usage
# -----------------------------------------------------------------------

def probe1_global_usage(model, hub, tokenizer, eval_dir, device, chunk_size=100_000):
    embedding = model.get_input_embeddings()
    N = hub.num_embeddings
    uniform_mass = 1.0 / N
    threshold = 0.1 * uniform_mass

    per_lang_mass = {}
    per_lang_tokens = {}

    for lang in LANGS:
        lang_dir = os.path.join(eval_dir, lang)
        if not os.path.isdir(lang_dir):
            print(f"  [{lang}] Not found, skipping.")
            continue

        ds = load_from_disk(lang_dir)
        texts = ds["text"]
        encodings = tokenizer("\n\n".join(texts), return_tensors="pt", add_special_tokens=False)
        input_ids = encodings.input_ids.squeeze(0)
        total_tokens = input_ids.shape[0]

        mass_acc = torch.zeros(N, dtype=torch.float64)
        token_count = 0

        for start in range(0, total_tokens, chunk_size):
            chunk_ids = input_ids[start:start + chunk_size].unsqueeze(0).to(device)
            with torch.no_grad():
                token_emb = embedding(chunk_ids)
            weights = get_anchor_weights(hub, token_emb)
            mass_acc += weights.squeeze(0).double().sum(dim=0).cpu()
            token_count += chunk_ids.shape[1]

        per_lang_mass[lang] = mass_acc / token_count
        per_lang_tokens[lang] = token_count
        print(f"  [{lang}] {token_count:,} tokens processed")

    # Per-language dead fraction
    per_lang_dead = {}
    per_lang_alive_set = {}
    for lang, mass in per_lang_mass.items():
        dead = (mass < threshold).sum().item()
        per_lang_dead[lang] = dead / N
        per_lang_alive_set[lang] = set((mass >= threshold).nonzero(as_tuple=True)[0].tolist())

    # Global mass (equal-weight since all 10M)
    global_mass = torch.zeros(N, dtype=torch.float64)
    total_global_tokens = 0
    for lang, mass in per_lang_mass.items():
        t = per_lang_tokens[lang]
        global_mass += mass * t
        total_global_tokens += t
    global_mass /= total_global_tokens

    global_dead_frac = (global_mass < threshold).sum().item() / N

    # Anchors dead in ALL languages
    dead_in_all = set(range(N))
    for lang, alive in per_lang_alive_set.items():
        dead_in_all -= alive
    dead_in_all_count = len(dead_in_all)

    # Anchors alive in ALL languages
    alive_in_all = None
    for lang, alive in per_lang_alive_set.items():
        if alive_in_all is None:
            alive_in_all = alive.copy()
        else:
            alive_in_all &= alive
    alive_in_all_count = len(alive_in_all) if alive_in_all else 0

    # Anchors alive in exactly ONE language
    anchor_lang_count = torch.zeros(N, dtype=torch.int32)
    for lang, alive in per_lang_alive_set.items():
        for a in alive:
            anchor_lang_count[a] += 1
    alive_in_one = (anchor_lang_count == 1).sum().item()

    # Pairwise Jaccard of alive sets
    jaccard = {}
    langs_with_data = list(per_lang_alive_set.keys())
    for i, l1 in enumerate(langs_with_data):
        for l2 in langs_with_data[i + 1:]:
            s1, s2 = per_lang_alive_set[l1], per_lang_alive_set[l2]
            j = len(s1 & s2) / len(s1 | s2) if len(s1 | s2) > 0 else 0
            jaccard[f"{l1}-{l2}"] = j

    # Global top10 mass
    global_top10 = global_mass.topk(10).values.sum().item()
    per_lang_top10 = {lang: mass.topk(10).values.sum().item() for lang, mass in per_lang_mass.items()}

    # Residual pairwise cosine (all anchors, not just first 100)
    A = hub.hub_embeddings.detach().float()
    mu = A.mean(0)
    res = A - mu
    Rn = F.normalize(res, dim=-1)
    cos_matrix = Rn @ Rn.T
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=A.device), diagonal=1)
    residual_cos = cos_matrix[mask].mean().item()

    return {
        "global_dead_frac": global_dead_frac,
        "global_top10_mass": global_top10,
        "dead_in_all_languages": dead_in_all_count,
        "alive_in_all_languages": alive_in_all_count,
        "alive_in_one_language": alive_in_one,
        "per_lang_dead": per_lang_dead,
        "per_lang_top10": per_lang_top10,
        "pairwise_jaccard": jaccard,
        "residual_pairwise_cos": residual_cos,
    }


# -----------------------------------------------------------------------
# PROBE 2 — Cross-lingual anchor overlap
# -----------------------------------------------------------------------

def get_word_anchor_dist(hub, embedding, tokenizer, word, device):
    """Get anchor attention distribution for a word (mean-pool if multi-token)."""
    ids = tokenizer(word, add_special_tokens=False, return_tensors="pt")["input_ids"]
    ids = ids.to(device)
    with torch.no_grad():
        token_emb = embedding(ids)
    weights = get_anchor_weights(hub, token_emb)
    return weights.squeeze(0).mean(dim=0)


def js_divergence(p, q):
    m = 0.5 * (p + q)
    eps = 1e-12
    kl_pm = (p * (p.clamp(min=eps) / m.clamp(min=eps)).log()).sum()
    kl_qm = (q * (q.clamp(min=eps) / m.clamp(min=eps)).log()).sum()
    return (0.5 * (kl_pm + kl_qm)).item()


def topk_jaccard(w1, w2, k=10):
    s1 = set(w1.topk(k).indices.tolist())
    s2 = set(w2.topk(k).indices.tolist())
    return len(s1 & s2) / len(s1 | s2) if len(s1 | s2) > 0 else 0


def probe2_crosslingual(hub, embedding, tokenizer, device, k=10):
    import random
    random.seed(42)

    # Get token counts per word
    word_dists = {}
    token_counts = {}
    for tuple_idx, tup in enumerate(TRANSLATION_TUPLES):
        for lang_idx, (lang, word) in enumerate(zip(LANGS, tup)):
            ids = tokenizer(word, add_special_tokens=False)["input_ids"]
            key = (tuple_idx, lang)
            token_counts[key] = len(ids)
            dist = get_word_anchor_dist(hub, embedding, tokenizer, word, device)
            word_dists[key] = dist

    # Report single-token stats
    single_token_per_lang = {lang: 0 for lang in LANGS}
    for (tuple_idx, lang), count in token_counts.items():
        if count == 1:
            single_token_per_lang[lang] += 1
    print(f"  Single-token words per language: {single_token_per_lang}")

    # Translation pair similarities (all cross-language pairs within each tuple)
    trans_jaccards = []
    trans_js_sims = []
    for tuple_idx in range(len(TRANSLATION_TUPLES)):
        for i, l1 in enumerate(LANGS):
            for l2 in LANGS[i + 1:]:
                w1 = word_dists.get((tuple_idx, l1))
                w2 = word_dists.get((tuple_idx, l2))
                if w1 is not None and w2 is not None:
                    trans_jaccards.append(topk_jaccard(w1, w2, k))
                    trans_js_sims.append(1.0 - js_divergence(w1, w2))

    # Random pair control (unrelated words, same language pairs)
    rand_jaccards = []
    rand_js_sims = []
    n_tuples = len(TRANSLATION_TUPLES)
    for _ in range(len(trans_jaccards)):
        t1, t2 = random.sample(range(n_tuples), 2)
        l1, l2 = random.sample(LANGS, 2)
        w1 = word_dists.get((t1, l1))
        w2 = word_dists.get((t2, l2))
        if w1 is not None and w2 is not None:
            rand_jaccards.append(topk_jaccard(w1, w2, k))
            rand_js_sims.append(1.0 - js_divergence(w1, w2))

    # Per language-pair breakdown
    per_pair = {}
    for i, l1 in enumerate(LANGS):
        for l2 in LANGS[i + 1:]:
            pair_key = f"{l1}-{l2}"
            pair_trans_j = []
            pair_trans_js = []
            for tuple_idx in range(n_tuples):
                w1 = word_dists.get((tuple_idx, l1))
                w2 = word_dists.get((tuple_idx, l2))
                if w1 is not None and w2 is not None:
                    pair_trans_j.append(topk_jaccard(w1, w2, k))
                    pair_trans_js.append(1.0 - js_divergence(w1, w2))
            if pair_trans_j:
                per_pair[pair_key] = {
                    "jaccard_mean": sum(pair_trans_j) / len(pair_trans_j),
                    "js_sim_mean": sum(pair_trans_js) / len(pair_trans_js),
                }

    # Significance test
    if trans_jaccards and rand_jaccards:
        stat_j, pval_j = stats.mannwhitneyu(trans_jaccards, rand_jaccards, alternative="greater")
        stat_js, pval_js = stats.mannwhitneyu(trans_js_sims, rand_js_sims, alternative="greater")
    else:
        pval_j = pval_js = 1.0

    # Example: "dog" tuple
    dog_idx = next((i for i, t in enumerate(TRANSLATION_TUPLES) if t[0] == "dog"), None)
    example = {}
    if dog_idx is not None:
        for lang_idx, lang in enumerate(LANGS):
            w = word_dists.get((dog_idx, lang))
            if w is not None:
                top_anchors = w.topk(k).indices.tolist()
                example[lang] = {"word": TRANSLATION_TUPLES[dog_idx][lang_idx], "top_anchors": top_anchors}

    return {
        "translation_jaccard_mean": sum(trans_jaccards) / len(trans_jaccards) if trans_jaccards else 0,
        "random_jaccard_mean": sum(rand_jaccards) / len(rand_jaccards) if rand_jaccards else 0,
        "jaccard_gap": (sum(trans_jaccards) / len(trans_jaccards) - sum(rand_jaccards) / len(rand_jaccards)) if trans_jaccards else 0,
        "jaccard_pvalue": pval_j,
        "translation_js_sim_mean": sum(trans_js_sims) / len(trans_js_sims) if trans_js_sims else 0,
        "random_js_sim_mean": sum(rand_js_sims) / len(rand_js_sims) if rand_js_sims else 0,
        "js_gap": (sum(trans_js_sims) / len(trans_js_sims) - sum(rand_js_sims) / len(rand_js_sims)) if trans_js_sims else 0,
        "js_pvalue": pval_js,
        "per_pair": per_pair,
        "example_dog": example,
        "single_token_per_lang": single_token_per_lang,
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EmbHub Anchor Probes")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="temp/anchor_probes_RESULTS.md")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoints[0])

    all_results = {}

    for ckpt in args.checkpoints:
        step = os.path.basename(ckpt).replace("checkpoint-", "")
        print(f"\n{'='*60}")
        print(f"  Checkpoint: {ckpt} (step {step})")
        print(f"{'='*60}")

        model, hub = load_checkpoint(ckpt, device=args.device)

        print(f"\n  PROBE 1 — Global anchor usage")
        p1 = probe1_global_usage(model, hub, tokenizer, args.eval_dir, args.device)

        print(f"\n  PROBE 2 — Cross-lingual overlap")
        embedding = model.get_input_embeddings()
        p2 = probe2_crosslingual(hub, embedding, tokenizer, args.device)

        all_results[step] = {"probe1": p1, "probe2": p2}

        del model, hub
        torch.cuda.empty_cache()

    # Write results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("# EmbHub Anchor Probe Results\n\n")

        for step, res in all_results.items():
            p1, p2 = res["probe1"], res["probe2"]

            f.write(f"## Checkpoint step {step}\n\n")

            # Probe 1
            f.write("### Probe 1 — Global & Per-Language Anchor Usage\n\n")
            f.write(f"- Global dead fraction: **{p1['global_dead_frac']:.3f}** ({p1['global_dead_frac']*1000:.0f}/1000)\n")
            f.write(f"- Global top-10 mass: **{p1['global_top10_mass']:.4f}**\n")
            f.write(f"- Dead in ALL languages: **{p1['dead_in_all_languages']}**/1000\n")
            f.write(f"- Alive in ALL languages: **{p1['alive_in_all_languages']}**/1000\n")
            f.write(f"- Alive in exactly ONE language: **{p1['alive_in_one_language']}**/1000\n")
            f.write(f"- Residual pairwise cosine: **{p1['residual_pairwise_cos']:.4f}**\n\n")

            f.write("Per-language dead fraction:\n\n")
            f.write("| Language | Dead Fraction | Top-10 Mass |\n")
            f.write("|----------|--------------|-------------|\n")
            for lang in LANGS:
                if lang in p1['per_lang_dead']:
                    f.write(f"| {lang} | {p1['per_lang_dead'][lang]:.3f} | {p1['per_lang_top10'].get(lang, 0):.4f} |\n")

            f.write("\nPairwise Jaccard (alive anchor sets):\n\n")
            f.write("| Pair | Jaccard |\n")
            f.write("|------|--------|\n")
            for pair, j in sorted(p1['pairwise_jaccard'].items()):
                f.write(f"| {pair} | {j:.3f} |\n")

            # Probe 2
            f.write(f"\n### Probe 2 — Cross-Lingual Anchor Overlap\n\n")
            f.write(f"Single-token words per language: {p2['single_token_per_lang']}\n\n")
            f.write(f"| Metric | Translation | Random | Gap | p-value |\n")
            f.write(f"|--------|------------|--------|-----|--------|\n")
            f.write(f"| Top-{10} Jaccard | {p2['translation_jaccard_mean']:.4f} | {p2['random_jaccard_mean']:.4f} | {p2['jaccard_gap']:.4f} | {p2['jaccard_pvalue']:.2e} |\n")
            f.write(f"| 1 - JS divergence | {p2['translation_js_sim_mean']:.4f} | {p2['random_js_sim_mean']:.4f} | {p2['js_gap']:.4f} | {p2['js_pvalue']:.2e} |\n")

            if p2['per_pair']:
                f.write(f"\nPer language-pair (Jaccard / JS similarity):\n\n")
                f.write("| Pair | Jaccard | JS Sim |\n")
                f.write("|------|---------|--------|\n")
                for pair, vals in sorted(p2['per_pair'].items()):
                    f.write(f"| {pair} | {vals['jaccard_mean']:.4f} | {vals['js_sim_mean']:.4f} |\n")

            if p2.get('example_dog'):
                f.write(f"\nExample — 'dog' top-10 anchors:\n\n")
                f.write("| Language | Word | Top-10 Anchors |\n")
                f.write("|----------|------|---------------|\n")
                for lang, info in p2['example_dog'].items():
                    f.write(f"| {lang} | {info['word']} | {info['top_anchors']} |\n")

            f.write("\n---\n\n")

    print(f"\nResults saved to {args.output}")

    # Also save raw JSON
    json_path = args.output.replace(".md", ".json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Raw data saved to {json_path}")


if __name__ == "__main__":
    main()
