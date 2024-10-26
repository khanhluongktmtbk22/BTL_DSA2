#include "main.hpp"
#include "Dataset.hpp"
/* TODO: Please design your data structure carefully so that you can work with the given dataset
 *       in this assignment. The below structures are just some suggestions.
 */
struct kDTreeNode
{
    int label;
    int distance;
    vector<int> data;
    kDTreeNode *left;
    kDTreeNode *right;
    kDTreeNode(vector<int> data, kDTreeNode *left = nullptr, kDTreeNode *right = nullptr, int label = -1, int distance = -1){
        this->data = data;
        this->left = left;
        this->right = right;
        this->label = label;
        this->distance = distance;
    }
    
    friend ostream &operator<<(ostream &os, const kDTreeNode &node)
    {
        os << "(";
        for (int i = 0; i < node.data.size(); i++)
        {
            os << node.data[i];
            if (i != node.data.size() - 1)
            {
                os << ", ";
            }
        }
        os << ")";
        return os;
    }
};

class kDTree
{
private:
    int k;
    kDTreeNode *root;

public:
    kDTree(int k = 2);
    ~kDTree();

    const kDTree &operator=(const kDTree &other);
    kDTree(const kDTree &other);

    void inorderTraversal() const;
    void preorderTraversal() const;
    void postorderTraversal() const;
    int height() const;
    int nodeCount() const;
    int leafCount() const;

    void insert(const vector<int> &point);
    void remove(const vector<int> &point);
    bool search(const vector<int> &point);
    void buildTree(const vector<vector<int>> &pointList);
    void buildTree_labled(const vector<vector<int>> &pointList, const vector<int> &labelList);
    void nearestNeighbour(const vector<int> &target, kDTreeNode *&best);
    void kNearestNeighbour(const vector<int> &target, int k, vector<kDTreeNode *> &bestList);
};

class kNN
{
private:
    int k;
    kDTree *tree;
public:
    kNN(int k = 5);
    ~kNN();
    void fit(Dataset &X_train, Dataset &y_train);
    Dataset predict(Dataset &X_test);
    double score(const Dataset &y_test, const Dataset &y_pred);
};

// Please add more or modify as needed
void reHeapUp(vector<kDTreeNode *> &heap, int index);
void reHeapDown(vector<kDTreeNode *> &heap, int index);
void rec_inorderTraversal(kDTreeNode *node, int k);
void rec_preorderTraversal(kDTreeNode *node, int k);
void rec_postorderTraversal(kDTreeNode *node, int k);
int rec_height(kDTreeNode *node);
int rec_nodeCount(kDTreeNode *node);
int rec_leafCount(kDTreeNode *node);
void rec_insert(kDTreeNode *&node, const vector<int> &point, int depth, int k);
bool compareVectors(const vector<int> &vec1, const vector<int> &vec2);
bool rec_search(kDTreeNode *node, const vector<int> &point, int depth, int k);
kDTreeNode *minNode(kDTreeNode *x, kDTreeNode *y, kDTreeNode *z, int axis);
kDTreeNode *findMin(kDTreeNode *node, int k, int axis, int depth);
void rec_remove(kDTreeNode *&node, const vector<int> &point, int depth, int k);
void merge(vector<vector<int>> &pointList, int l, int m, int r, int axis);
void mergeSort(vector<vector<int>> &pointList, int l, int r, int axis);
void rec_buildTree(kDTreeNode *&node, vector<vector<int>> &pointList, int k, int depth);
void merge_labled(vector<vector<int>> &pointList, vector<int> &labelList, int l, int m, int r, int axis);
void mergeSort_labled(vector<vector<int>> &pointList, vector<int> &labelList, int l, int r, int axis);
void rec_buildTree_labled(kDTreeNode *&node, vector<vector<int>> &pointList, vector<int> &labelList, int k, int depth);
int distance(const vector<int> &point1, const vector<int> &point2);
void rec_nearestNeighbour(kDTreeNode *node, const vector<int> &target, kDTreeNode *&best, int depth, int k);
void rec_kNearestNeighbour(kDTreeNode *node, const vector<int> &target, int k, vector<kDTreeNode *> &bestList, int depth, int d);
void sortBestList(vector<kDTreeNode *> &bestList, const vector<int> &target);


