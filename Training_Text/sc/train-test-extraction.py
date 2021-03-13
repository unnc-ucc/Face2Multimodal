f = open("/Users/billxzy1215/Desktop/Research/Face/Model-Training/sc/p2.txt")
train = open("/Users/billxzy1215/Desktop/Research/Face/Model-Training/sc/train2.txt","w")
test = open("/Users/billxzy1215/Desktop/Research/Face/Model-Training/sc/test2.txt","w")
scs = []
for line in f.readlines():
    sc = line.split("\n")[0]
    scs.append(sc)

count = 0
for i in scs:
    print(i)
    count = count + 1
    if(count % 5 == 0):
        test.write(i+"\n")
    else:
        train.write(i+"\n")

f.close()
train.close()
test.close()