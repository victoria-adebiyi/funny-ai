files = ["train", "test"]

for file in files:
    with open('processed/' + file + '.ssv', mode='w') as w:
        w.write("score;;;;;post\n")

        with open('rJokesData/data/' + file + '.tsv') as r:
            for line in r:
                if line[1] == "	":
                    w.write(line[0] + ";;;;;" + line[2:])
                else:
                    w.write(line[0] + line[1] + ";;;;;" + line[3:])