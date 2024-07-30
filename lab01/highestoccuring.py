def highestoccuring(str):
    char_count = {}

    for c in str:
        c = c.lower()
        if c.isalpha():
            if not char_count.get(c):
                char_count[c] = 1
            else:
                char_count[c] += 1

    max_char = None
    max_count = 0
    for i in char_count.keys():
        if(char_count.get(i) > max_count):
            max_char = i
            max_count = char_count.get(i)

    return (max_char, max_count)

if __name__ == "__main__":
    result = highestoccuring("hippopotamus")
    print(f"{result[0]}: {result[1]}")