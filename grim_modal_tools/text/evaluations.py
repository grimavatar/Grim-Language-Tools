import math
import difflib
from .utils import classify_text_end


def round_secs(value: float) -> float:
    return int(round(value * 1000)) / 1000


def get_upper_mean(xs: list[float]) -> float:
    values = sorted(xs)[len(xs)//2:]
    return round_secs(sum(values) / len(values)) if values else math.nan


# def get_lower_mean(xs: list[float]) -> float:
#     values = sorted(xs)[:math.ceil(len(xs)/2)]
#     return round_secs(sum(values) / len(values)) if values else math.nan


def evaluate_alignment(alignment: list[dict], max_pause_allowed: float = 0.5, max_clause_allowed: float = 1.0, max_sent_allowed: float = 1.5) -> bool:
    no_punc_silence = False
    pause_silence = []
    clause_silence = []
    sent_silence = []
    for i in range(len(alignment)-1):
        part = alignment[i]
        part_n = alignment[i+1]
        diff = round_secs(part_n["start"] - part["end"])
        text = part["word"]
        # print(repr(text))

        is_pause, is_clause, is_sent = classify_text_end(text)
        if diff <= 0 and not is_pause:
            no_punc_silence = True
            break
        elif diff > 0:
            if is_pause:
                pause_silence.append(diff)
            elif is_clause:
                clause_silence.append(diff)
            elif is_sent:
                sent_silence.append(diff)
            # print("---------------", diff)

    if no_punc_silence:
        print("REJECT: no punc silence detected")
        return False

    max_pause_silence = max(pause_silence) if pause_silence else math.nan
    max_clause_silence = max(clause_silence) if clause_silence else math.nan
    max_sent_silence = max(sent_silence) if sent_silence else math.nan

    if max_pause_silence >= max_pause_allowed:
        print("REJECT: max pause silence reached")
        return False

    if max_clause_silence >= max_clause_allowed:
        print("REJECT: max clause silence reached")
        return False

    if max_sent_silence >= max_sent_allowed:
        print("REJECT: max sent silence reached")
        return False

    pause_upper_mean = get_upper_mean(pause_silence)
    clause_upper_mean = get_upper_mean(clause_silence)
    sent_upper_mean = get_upper_mean(sent_silence)

    if pause_upper_mean >= clause_upper_mean:
        print(pause_upper_mean, clause_upper_mean)
        print("REJECT: p_max(no_punc) >= p_max(small_medium)")
        return False

    if clause_upper_mean >= sent_upper_mean:
        print(clause_upper_mean, sent_upper_mean)
        print("REJECT: p_max(small_medium) >= p_max(long)")
        return False

    print(max_pause_silence, max_clause_silence, max_sent_silence)
    print(pause_upper_mean, clause_upper_mean, sent_upper_mean)
    return True


### <---Maybe-Useless--->

def make_script(text: str, norm: bool = False) -> str:
    text = "".join(e if e.isalnum() else " " for e in text)
    if norm:
        text = text.lower()
    return " ".join(text.split())

def make_ref(text: str) -> str:
    return "".join(e for e in text.lower() if e.isalnum())

def compare_texts(src_text: str, tgt_text: str) -> bool:
    src_text = make_ref(src_text)
    tgt_text = make_ref(tgt_text)
    return src_text == tgt_text

### <---Maybe-Useless--->


# def pass_asr_test(src_text: str, tgt_text: str, alignment: list[dict]) -> bool:
#     src_text = src_text.replace("—", "; ")
#     is_same = compare_texts(src_text, tgt_text)
#     if not is_same:
#         return False
#     # src_words = src_text.split()
#     # tgt_words = tgt_text.split()
#     # if len(src_words) != len(tgt_words):
#     #     src_words = src_text.replace("-", " ").split()
#     #     if len(src_words) != len(tgt_words):
#     #         print(src_words)
#     #         print(tgt_words)
#     #         return False
#     # alignment = deepcopy(alignment)
#     # assert tgt_words == [e["word"] for e in alignment]
#     # for i, part in enumerate(src_words):
#     #     alignment[i]["word"] = part
#     # is_aligned = evaluate_alignment(alignment)
#     # if not is_aligned:
#     #     return False
#     return True


# https://github.com/jitsi/jiwer
# https://github.com/emorynlp/align4d
def _align_to_source_v1(src_text: str, alignment: list[dict]) -> list[dict] | None:
    def normalize(w: str) -> str:
        return "".join(c for c in w.lower() if c.isalnum())

    for i in range(len(alignment)-1, -1, -1):
        if not alignment[i]["word"].strip():
            alignment.pop(i)

    src_words = src_text.split()
    tgt_words = [e["word"] for e in alignment]
    src_n = [normalize(w) for w in src_words]
    tgt_n = [normalize(w) for w in tgt_words]

    if "".join(src_n) != "".join(tgt_n):
        return None

    result = []
    i = j = 0
    while i < len(src_n) and j < len(tgt_n):
        si, sj = i, j
        s_acc, t_acc = src_n[i], tgt_n[j]

        while s_acc != t_acc:
            if len(s_acc) <= len(t_acc):
                i += 1
                if i >= len(src_n):
                    return None
                s_acc += src_n[i]
            else:
                j += 1
                if j >= len(tgt_n):
                    return None
                t_acc += tgt_n[j]

        result.append({
            "word": " ".join(src_words[si:i + 1]),
            # "start_offset": alignment[sj]["start_offset"],
            # "end_offset": alignment[j]["end_offset"],
            "start": alignment[sj]["start"],
            "end": alignment[j]["end"],
        })
        i += 1
        j += 1
    
    val_words = [e["word"] for e in result]
    if val_words != src_words:
        raise ValueError("Implementation error.")

    return result


def _align_to_source_v2(src_text: str, alignment: list[dict], length_error: int = 1, max_errors: int = 1) -> list[dict] | None:

    def is_strong_overlap(a: str, b: str, length_error: int) -> bool:
        return (
            a.startswith(b) or a.endswith(b) or
            b.startswith(a) or b.endswith(a)
        ) and abs(len(a) - len(b)) <= length_error

    def normalize(w: str) -> str:
        text = "".join(c if c.isalnum() else " " for c in w.lower())
        return " ".join(text.split())

    alignment = [e for e in alignment if e["word"].strip()]
    src_words = src_text.split()
    tgt_words = [e["word"] for e in alignment]

    src_norm = [normalize(w) for w in src_words]
    tgt_norm = [normalize(w) for w in tgt_words]
    # print(src_norm)
    # print(tgt_norm)

    # Word-level diff on normalized forms
    matcher = difflib.SequenceMatcher(None, src_norm, tgt_norm)
    result = []
    errors = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        # print(tag, i1, i2, j1, j2)

        # If counts match, assume ordered 1:1 mapping.
        # Otherwise, matcherear the whole block's time across the source words.
        is_1_to_1 = (j2 - j1) == (i2 - i1)
        if j1 == j2:  # delete event
            start = end = result[-1]["end"] if result else 0.0
        else:
            start = alignment[j1]["start"]
            end = alignment[j2 - 1]["end"]

        src_group = []
        tgt_group = alignment[j1:j2]
        for offset, si in enumerate(range(i1, i2)):
            tj = j1 + offset if j1 + offset < j2 else j2 - 1
            src_group.append({
                "word": src_words[si],
                "start": alignment[tj]["start"] if is_1_to_1 else start,
                "end": alignment[tj]["end"] if is_1_to_1 else end,
            })
        # print(src_group)
        # print(tgt_group)
        # print("---")

        src_terms = [w for e in src_group for w in normalize(e["word"]).split()]
        tgt_terms = [w for e in tgt_group for w in normalize(e["word"]).split()]
        # print(src_terms)
        # print(tgt_terms)
        # print("---")

        if len(src_terms) == len(tgt_terms):
            for src_t, tgt_t in zip(src_terms, tgt_terms):
                if src_t != tgt_t:
                    errors += 1
                # print(errors)
                if not (errors <= max_errors and is_strong_overlap(src_t, tgt_t, length_error)):
                    return None
        else:
            return None

        result.extend(src_group)

    # Sanity check
    if [e["word"] for e in result] != src_words:
        raise ValueError("Alignment produced unexpected word order")

    return result


def align_to_source(src_text: str, alignment: list[dict], length_error: int = 1, max_errors: int = 1, stable: bool = False) -> list[dict] | None:
    if stable:
        return _align_to_source_v1(src_text, alignment)
    else:
        return _align_to_source_v2(src_text, alignment, length_error, max_errors)


def align_to_source_w_test(src_text, alignment, audio_path):
    import orjson
    from pathlib import Path
    alignment_v1 = align_to_source(src_text, alignment, stable = True)
    alignment_v2 = align_to_source(src_text, alignment, stable = False)
    if alignment_v1 != alignment_v2:
        output_path = Path(audio_path).with_suffix(".json")
        output_path = output_path.parent / f"{output_path.parent.name}_{output_path.name}"
        with open(output_path, "wb") as f:
            data = {"src_text": src_text, "alignment": alignment}
            f.write(orjson.dumps(data))
    return alignment_v2


def pass_asr_test(src_text: str, alignment: list[dict]) -> bool:
    src_text = src_text.replace("—", "; ")    
    alignment = align_to_source(src_text, alignment)
    if alignment is None:
        return False

    # is_aligned = evaluate_alignment(alignment)
    # if not is_aligned:
    #     return False
    
    return True
