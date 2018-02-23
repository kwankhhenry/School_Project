#include <iostream>
#include "BST.h"

using namespace std;

int main()
{
	int TreeKeys[16] = {50,76,21,4,32,64,15,52,14,100,83,2,3,70,87,80};

	BST myTree;

	cout << "Printing the tree in order\nBefore adding numbers\n";

	myTree.PrintInOrder();

	for(int i = 0; i < 16; i++)
	{
		myTree.AddLeaf(TreeKeys[i]);
	}

	cout << "Printing the tree in order\nAfter adding numbers\n";

	myTree.PrintInOrder();

	cout << endl;

	cout << "The smalleset value in the tree is " << myTree.FindSmallest();

	cout << endl;

	myTree.Print_Tree_Level(3);

	cout << endl;

	return 0;
}
