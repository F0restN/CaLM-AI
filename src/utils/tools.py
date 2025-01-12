from logger import logger

class ToolKits:
    def __init__(self):
        pass

    def straight_array_to_string(self, array: list, separator: str = ",") -> str:
        if not array:
            return "NAN"
        return separator.join(array)
    


if __name__ == "__main__":
    tools = ToolKits()
    print(tools.straight_array_to_string([]))
