#ifndef BST_H
#define BST_H

class BST{

		private:
			struct node{
				int key;
				node* left;
				node* right;
			};

			node* root;

			void AddLeafPrivate(int key, node* Ptr);
			void PrintInOrderPrivate(node* Ptr);
			node* ReturnNodePrivate(int key, node* Ptr);
			int FindSmallestPrivate(node* Ptr);
			void Print_Tree_Level_Private(node* root, int depth);

		public:
			BST();
			node* CreateLeaf(int key);
			void AddLeaf(int key);

			void PrintInOrder();
			node* ReturnNode(int key);
			void PrintChildren(int key);
			int ReturnRootKey();
			int FindSmallest();

			void Print_Tree_Level(int depth);

};

#endif
