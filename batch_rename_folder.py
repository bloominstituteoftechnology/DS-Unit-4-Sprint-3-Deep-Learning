# Python 3 code to rename multiple files in folder

import os
def main():
	os.chdir(r"C:\Users\benjamin\Desktop\DS-Unit-4-Sprint-3-Deep-Learning\module2-convolutional-neural-networks\data\forest")
	i = 0
	for filename in os.listdir('C:'):
		dst = "forest" + str(i) + ".jpg"
		src = 'C:' + filename
		dst = 'C:' + dst
		os.rename(src, dst)
		i += 1
	os.chdir(r"C:\Users\benjamin\Desktop\DS-Unit-4-Sprint-3-Deep-Learning\module2-convolutional-neural-networks\data\mountain")
	i = 0
	for filename in os.listdir('C:'):
		dst = "mountain" + str(i) + ".jpg"
		src = 'C:' + filename
		dst = 'C:' + dst
		os.rename(src, dst)
		i += 1



if __name__ == '__main__':
	main()