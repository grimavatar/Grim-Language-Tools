import math
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
def align_to_source(src_text: str, alignment: list[dict]) -> list[dict] | None:
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


def pass_asr_test(src_text: str, alignment: list[dict]) -> bool:
    src_text = src_text.replace("—", "; ")    
    alignment = align_to_source(src_text, alignment)
    if alignment is None:
        return False

    # is_aligned = evaluate_alignment(alignment)
    # if not is_aligned:
    #     return False
    
    return True
