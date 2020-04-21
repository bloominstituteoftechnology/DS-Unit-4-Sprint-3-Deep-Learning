i = 0
try:
    with open('100-0.txt', 'r') as f:
        WORKS = []
        for l in f:
            if i > 42:
                if l.strip() != "":
                    WORKS.append(l.strip())
            elif i > 129:
                break
            else:
                i += 1
except:
    print(i)


smallfile = None
with open('100-0.txt') as bigfile:
    for lineno, line in enumerate(bigfile):
        if line.strip().replace("’", "'") in WORKS:
            if smallfile:
                smallfile.close()
            small_filename = f'{line.strip()}.txt'
            smallfile = open(small_filename, "w")
        if smallfile:
            smallfile.write(line.replace("’", "'"))
    if smallfile:
        smallfile.close()
