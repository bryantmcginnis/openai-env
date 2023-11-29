"""
Splits a text string into chunks of `max_length` characters or less.
Tries to split at paragraph breaks first, then at spaces to avoid word splits.

:param text: The text to split.
:param max_length: Maximum length of each chunk. Default is 2000.
:return: A list of text chunks.
"""
class ResponseSplitter:
    def split(self, text, max_length=2000):

        chunks = []
        while text:
            if len(text) <= max_length:
                chunks.append(text)
                break

            # Try to split at the nearest previous paragraph break
            split_index = text.rfind("\n\n", 0, max_length)

            # If no paragraph break, try to split at the nearest space
            if split_index == -1:
                split_index = text.rfind(" ", 0, max_length)

            # If no space is found, force split at max_length
            if split_index == -1:
                split_index = max_length

            chunks.append(text[:split_index].strip())
            text = text[split_index:].strip()

        return chunks