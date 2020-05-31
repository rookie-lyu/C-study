[TOC]

##### 5.28 470 96

```c++
int rand10()
{
	int col,row,idx;
	do
	{
		col=rand7();
		row=rand7();
		idx=(row-1)*7+col;
	}while(idx>40)
	return (idx-1)%10+1;
}
```

```C++
int numTrees(int n) 
{
    vector<int>dp(n+1,0);
    dp[0]=1;
    for(int i=1;i<=n;i++)
    {
        for(int j=1;j<=i;j++)
        {
            dp[i]+=dp[j-1]*dp[i-j];
        }
    }
    return dp[n];
}
```

##### 5.29 64 94 19

```c++
int minPathSum(vector<vector<int>>& grid) 
    {
     int n = grid.size();
        if(n==0)
            return 0;
        int m = grid[0].size();
        if(m==0)
            return 0;
        for(int i=1;i<m;i++)//第一行最短路径
        {
            grid[0][i] += grid[0][i-1];

        }
        for(int i=1;i<n;i++)//第一列最短路径
        {
            grid[i][0] += grid[i-1][0];
        }
        for(int i=1;i<n;i++)
        {
            for(int j=1;j<m;j++)
            {
                grid[i][j]+=min(grid[i-1][j],grid[i][j-1]);
            }
        }
        return grid[n-1][m-1];

        
    }
```

```c++
vector<int> inorderTraversal(TreeNode* root) {
        vector<int> ans;
        stack<TreeNode*> stk;
        TreeNode* temp = root;
        while(temp || !stk.empty()){
            while(temp)
            {
                stk.push(temp);
                temp=temp->left;
            }
            temp=stk.top();
            stk.pop();
            ans.push_back(temp->val);
            temp=temp->right;
        }
        return ans;
   

    }
```

19.给定一个链表，删除链表的倒数第 *n* 个节点，并且返回链表的头结点。

```C++
 ListNode* removeNthFromEnd(ListNode* head, int n)
    {
        if(head==NULL)
        return NULL;
        ListNode* dummy=new ListNode(NULL);
        dummy->next=head;
        ListNode *slow =dummy;
        ListNode  *fast=dummy;
        for(int i=1;i<=n;i++)
        fast=fast->next;
        while(fast->next!=NULL)
        {
            fast=fast->next;
            slow=slow->next;
        }
        
       slow->next=slow->next->next;

    return dummy->next;
    }
```



<span style='color:red;'>设置头结点要不会出现溢出</span>

##### 5.31 226 23

```C++
TreeNode* invertTree(TreeNode* root) {
        if(root==NULL)
        return NULL;
        TreeNode* tmp=root->left;
        root->left=root->right;
        root->right=tmp;
        if(root->left)
        invertTree(root->left);
        if(root->right)
        invertTree(root->right);
        return root;
    }
```

```c++
struct cmp{  
       bool operator()(ListNode *a,ListNode *b){
          return a->val > b->val;}
       };
//priority_queue自定义函数的比较与sort正好是相反的，也就是说，如果你是把大于号作为第一关键字的比较方式，那么堆顶的元素就是第一关键字最小的


    ListNode* mergeKLists(vector<ListNode*>& lists) {
          priority_queue<ListNode*, vector<ListNode*>, cmp> pri_queue;
        // 建立大小为k的小根堆
        for(auto elem : lists){
            if(elem) pri_queue.push(elem);
        }
        // 可以使用哑节点/哨兵节点
        ListNode dummy(-1);
        ListNode* p = &dummy;
        // 开始出队
        while(!pri_queue.empty()){
            ListNode* top = pri_queue.top(); pri_queue.pop();
            p->next = top; p = top;
            if(top->next) pri_queue.push(top->next);
        }
        return dummy.next; 


    }
```

