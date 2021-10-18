### this file is created to test some funcionality of the code


from os import listdir
from os.path import isfile, join
import re
import itertools
import networks
mypath = "/Users/andreferdinand/Desktop/Coding2/output/weights_log"
onlyfiles = [f for f in listdir() if isfile(join(mypath, f))]



if __name__ == "__main__":
    mypath = "/Users/andreferdinand/Desktop/Coding2/output/weights_log"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    r = re.compile("\d+")
    newlist = list(map(r.findall, onlyfiles))
    merged = list(itertools.chain(*newlist))
    merged = list(map(int,merged))
    merged.sort()
    print(merged)