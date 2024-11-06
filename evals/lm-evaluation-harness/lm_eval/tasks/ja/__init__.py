import re


class MecabTokenizer:
    def __init__(self) -> None:
        from fugashi import Tagger

        self.tagger = Tagger("-Owakati")

    def normalize_answer(self, text):
        """Lower case text, remove punctuation and extra whitespace, etc."""
        import emoji
        import neologdn

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_emoji(text):
            text = "".join(["" if emoji.is_emoji(c) else c for c in text])
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                "\U00002702-\U000027B0"
                "]+",
                flags=re.UNICODE,
            )
            return emoji_pattern.sub(r"", text)

        text = remove_emoji(text)
        # see neologdn docs for details, but handles things like full/half width variation
        text = neologdn.normalize(text)
        text = white_space_fix(text)
        return text

    def tokenize(self, text):
        return self.tagger.parse(self.normalize_answer(text)).split()
