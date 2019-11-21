'''import csv
with open('disease.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['Word', 'length','preweight','ratio','value'])
    for i in range(0,50):
        print(i)
        a=input("enter word")
        org=a
        leng=len(a)
        a=a[::-1]
        arr=a[0:(len(a)//3)+1]
        weig=0
        for i in range(0,len(arr)):
            weig=weig+ord(arr[i])
        val=input("enter value")
        tsv_writer.writerow([org,leng,weig,weig/leng,val])
        
    
    
