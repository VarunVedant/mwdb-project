"""
Phase 1 Project's main code
"""
import sys
import task1
import task2
import task3

def main():
	while True:
		print('Choose which task u want to execute: \n(Note - Execute Task 2 before Task 3)')
		ch = input('\n1. Task 1\n2. Task 2\n3. Task 3\n4. Exit\nEnter Choice: ')
		if ch == '1':
			task1.task1()
		elif ch == '2':
			task2.task2()
		elif ch == '3':
			task3.task3()
		elif ch == '4':
			break
pass

if __name__ == '__main__':
	sys.exit(main())
