[TOC]

##### 二分查找

```C++
 int findLowerBound(vector<int> &nums, int target) 
    {
        int size = nums.size();
        int left = 0;
        int right = size - 1;
        while (left < right) 
        {
            int mid = (left + right) >> 1;
            if (nums[mid] < target) 
            {
                left = mid + 1;
            } 
            else 
            {
                right = mid;
            }
        }
        if (nums[left] != target) {
            return -1;
        }
        return left;
    }

    int findUpBound(vector<int> &nums, int target) {
        int size = nums.size();
        int left = 0;
        int right = size - 1;
        while (left < right) {
            int mid = (left + right + 1) >> 1;
            if (nums[mid] > target) {
                right = mid - 1;
            } else {
                left = mid;

            }
        }
        if (nums[left] != target) {
            return -1;
        }
        return left;
    }
```



##### 5.28 470 96C++5.28 470 96

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

##### 6.1 33 42


假设按照升序排序的数组在预先未知的某个点上进行了旋转。

( 例如，数组 `[0,1,2,4,5,6,7]` 可能变为 `[4,5,6,7,0,1,2]` )。

搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 `-1` 。

你可以假设数组中不存在重复的元素。

你的算法时间复杂度必须是 *O*(log *n*) 级别。

```
输入: nums = [4,5,6,7,0,1,2], target = 0
输出: 4
```

```
输入: nums = [4,5,6,7,0,1,2], target = 3
输出: -1
```



```C++
int search(vector<int>& nums, int target)
    {
        int l = 0, r = nums.size() -1;
        
        while (l <= r)
        {
            int mid = (l + r) >> 1;
            if (target == nums[mid]) return mid;

            if (nums[l] <= nums[mid])//mid左半部分有序
            {
                if (target >= nums[l] && target < nums[mid])//在有序部分
                    r = mid-1;
                else
                    l = mid+1;
            }
            else//mid右半部分有序
            {
                if (target > nums[mid] && target <= nums[r])
                    l = mid +1;
                else
                    r = mid -1;
            }
        }
        return -1;
        
    }
```

42

思路：每一层的块相加，然后减去柱子的高度。

```
 int trap(vector<int>& height) 
    {   int left=0;int right=height.size()-1;
        int sum=0,tmp=0,heigh=1;
        while(left<=right)
        {
            while(left<=right && height[left]<heigh)
            {
                sum+=height[left];
                left++;
            }
            while(left<=right && height[right]<heigh)
            {
                sum+=height[right];
                right--;
            }
            heigh++;
            tmp+=right-left+1;
            
        }
        return tmp-sum;
    }
```

https://leetcode-cn.com/problems/trapping-rain-water/solution/shuang-zhi-zhen-an-xing-qiu-geng-hao-li-jie-onsuan/另一种双指针的写法

```C++
/以最大值分界，左边非减，右边非增
    int trap(vector<int>& height) {
        int n=height.size();
        if(n==0) return 0;
        int m=max_element(height.begin(),height.end())-height.begin();
        //遍历最大值左边
        int res=0,cur=height[0];
        for(int i=1;i<m;i++)
        {
            if(height[i]<cur)
                res+=cur-height[i];
            else
                cur=height[i];
        }
        //遍历最大值右边
        cur=height[n-1];
        for(int i=n-2;i>m;i--)
        {
            if(height[i]<cur)
                res+=cur-height[i];
            else
                cur=height[i];
        }
        return res;
    }
```

