import glob
a = glob.glob("vn/*ttf")
aa = glob.glob("vn/*TTF")
# glob,glob()
file1 = open("fontlist.txt","w")
for i in a:
    file1.writelines(i +"\n")
for i in aa:
    file1.writelines(i +"\n")
    # print(i)
    # print ("vn/"+a[i])
file1.close()
print(glob.glob("*ttf"))
