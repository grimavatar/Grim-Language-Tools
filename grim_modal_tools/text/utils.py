import os


# from spacy.lang.en.tokenizer_exceptions import TOKENIZER_EXCEPTIONS as PUNC_END_EXCEPTIONS
PUNC_END_EXCEPTIONS = ["a.", "b.", "c.", "d.", "e.", "f.", "g.", "h.", "i.", "j.", "k.", "l.", "m.", "n.", "o.", "p.", "q.", "r.", "s.", "t.", "u.", "v.", "w.", "x.", "y.", "z.", "ä.", "ö.", "ü.", "._.", "°c.", "°f.", "°k.", "1a.m.", "1p.m.", "2a.m.", "2p.m.", "3a.m.", "3p.m.", "4a.m.", "4p.m.", "5a.m.", "5p.m.", "6a.m.", "6p.m.", "7a.m.", "7p.m.", "8a.m.", "8p.m.", "9a.m.", "9p.m.", "10a.m.", "10p.m.", "11a.m.", "11p.m.", "12a.m.", "12p.m.", "mt.", "ak.", "ala.", "apr.", "ariz.", "ark.", "aug.", "calif.", "colo.", "conn.", "dec.", "del.", "feb.", "fla.", "ga.", "ia.", "id.", "ill.", "ind.", "jan.", "jul.", "jun.", "kan.", "kans.", "ky.", "la.", "mar.", "mass.", "mich.", "minn.", "miss.", "n.c.", "n.d.", "n.h.", "n.j.", "n.m.", "n.y.", "neb.", "nebr.", "nev.", "nov.", "oct.", "okla.", "ore.", "pa.", "s.c.", "sep.", "sept.", "tenn.", "va.", "wash.", "wis.", "a.m.", "adm.", "bros.", "co.", "corp.", "d.c.", "dr.", "e.g.", "gen.", "gov.", "i.e.", "inc.", "jr.", "ltd.", "md.", "messrs.", "mo.", "mont.", "mr.", "mrs.", "ms.", "p.m.", "ph.d.", "prof.", "rep.", "rev.", "sen.", "st.", "vs.", "v.s."]
TRAILING_CLOSERS = "\"'”’)]}"


def classify_text_end(text: str) -> tuple[bool, bool, bool]:
    """Returns tuple[is_pause, is_clause, is_sent]"""
    if not text or text[-1].isalnum():
        return True, False, False
    last_word = text.split()[-1].rstrip(TRAILING_CLOSERS).lower()
    if not last_word or last_word in PUNC_END_EXCEPTIONS:
        return True, False, False

    last_char = last_word[-1]
    if last_char in ".?!":
        return False, False, True
    else:
        return False, True, False


def segment_text(text: str, min_units: int = 2) -> list[str]:
    text = text.replace(" —", "—").replace("—", "— ")
    segments, units = [], []
    for word in text.split():
        units.append(word)
        _, is_clause, is_sent = classify_text_end(word)
        if is_sent or (is_clause and len(units) >= min_units):
            segments.append(" ".join(units))
            units = []
    return segments


def normalize_spaces(text: str) -> str:
    return "".join(" " if e.isspace() and e != "\n" else e for e in text.strip())


def normalize_punctuation(text: str) -> str:
    """
    Convert Unicode punctuation marks to ASCII equivalents.
    """
    # Mapping of Unicode punctuation to ASCII punctuation
    uni_to_ascii_punct = {
        "，": ", ", # comma
        "。": ". ",  # period
        "：": ": ",  # colon
        "；": "; ",  # semicolon
        "？": "? ",  # question mark
        "！": "! ",  # exclamation mark
        "（": " (",  # left parenthesis
        "）": ") ",  # right parenthesis
        "【": " [",  # left square bracket
        "】": "] ",  # right square bracket
        "《": " <",  # left angle quote
        "》": "> ",  # right angle quote
        "“": '"',   # left double quotation
        "”": '"',   # right double quotation
        "‘": "'",   # left single quotation
        "’": "'",   # right single quotation
        "、": ",",  # enumeration comma
        " — ": "—", # em dash
        " —": "—",  # em dash
        "— ": "—",  # em dash
        "–": "-",   # dash
        "…": "...", # ellipsis
        "·": ".",   # middle dot
        "「": ' "',  # left corner bracket
        "」": '" ',  # right corner bracket
        "『": ' "',  # left double corner bracket
        "』": '" ',  # right double corner bracket

        "&": 'and',  # right double corner bracket

        # "—": " — ",  # em dash
        # "—": ", ",  # em dash
        "—": "; ",  # em dash
        "##": "",
        " (": ", ",
        ") ": ", ",
        " [": ", ",
        "] ": ", ",
        # " <": ", ",
        # "> ": ", ",

        "°F": " degrees Fahrenheit",
        "°C": " degrees Celsius",

    }

    # Replace each Unicode punctuation with its ASCII counterpart
    for uni_punct, ascii_punct in uni_to_ascii_punct.items():
        text = text.replace(uni_punct, ascii_punct)

    return text


def normalize_text(text: str, fix_punc: bool = True) -> str:
    text = normalize_spaces(text)  # First-pass
    
    if fix_punc and not any([text.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'"]]):
        text += "."
    
    text = normalize_punctuation(text)
    text = normalize_spaces(text)  # Second-pass

    return text


def prepare_text(
    text: str,
    fix_punc: bool = True,
):
    if os.path.exists(text):
        with open(text, "rt", encoding = "utf-8") as f:
            text = f.read().strip()

    split_text = lambda text: [x for e in text.split("\n") if (x := e.strip())]
    sections = [x for e in text.split("##") if (x := e.strip())]

    texts = []
    for section in sections:
        lines = split_text(section)
        for i, line in enumerate(lines):
            lines[i] = normalize_text(line, fix_punc = fix_punc)
        texts.append(lines)
    
    return texts
