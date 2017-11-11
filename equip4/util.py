def get_emoticones():
    with open("../ressrc/emoticones.txt") as f:
        for i in range(8):
            next(f)
        content = f.readlines()
    # to remove whitespace characters like `\n` at the end of each line
    return [x.strip() for x in content if x.strip()]
