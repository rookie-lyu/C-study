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

##### 排序

```C++
// 递归方式构建大根堆(len是arr的长度，index是第一个非叶子节点的下标)
void adjust(vector<int> &arr, int len, int index)
{
	int left = 2 * index + 1; // index的左子节点
	int right = 2 * index + 2;// index的右子节点

	int maxIdx = index;
	if (left<len && arr[left] > arr[maxIdx])     maxIdx = left;
	if (right<len && arr[right] > arr[maxIdx])     maxIdx = right;

	if (maxIdx != index)
	{
		swap(arr[maxIdx], arr[index]);
		adjust(arr, len, maxIdx);
	}

}

// 堆排序
void heapSort(vector<int> &arr, int size)
{
	// 构建大根堆（从最后一个非叶子节点向上）
	for (int i = size / 2 - 1; i >= 0; i--)
	{
		adjust(arr, size, i);
	}

	// 调整大根堆
	for (int i = size - 1; i >= 1; i--)
	{
		swap(arr[0], arr[i]);           // 将当前最大的放置到数组末尾
		adjust(arr, i, 0);              // 将未完成排序的部分继续进行堆排序
	}
}
void quicksort(vector<int> &arr,int low,int high)
{
	if (low < high)
	{
		int pivot = arr[low]; int i = low;int  j = high;
		while (i < j)
		{
			while (i< j&&arr[j] >= pivot)
				j--;
			swap(arr[i], arr[j]);
			while (i< j&&arr[i] <= pivot)
				i++;
			swap(arr[i], arr[j]);
		}
		arr[i] = pivot;
		quicksort(arr, low, i - 1);
		quicksort(arr, i + 1, high);
	}
}
void MergeSort(vector<int>arr, int low, int high) {
	if (low >= high) { return; } // 终止递归的条件，子序列长度为1
	int mid = low + (high - low) / 2;  // 取得序列中间的元素
	MergeSort(arr, low, mid);  // 对左半边递归
	MergeSort(arr, mid + 1, high);  // 对右半边递归
	merge(arr, low, mid, high);  // 合并
}
void merge(vector<int>arr, int low, int mid, int high){
	//low为第1有序区的第1个元素，i指向第1个元素, mid为第1有序区的最后1个元素
	int i = low, j = mid + 1, k = 0;  //mid+1为第2有序区第1个元素，j指向第1个元素
	//int *temp = new int[high - low + 1]; //temp数组暂存合并的有序序列
	vector<int>temp;
	while (i <= mid&&j <= high){
		if (arr[i] <= arr[j]) //较小的先存入temp中
			temp[k++] = arr[i++];
		else
			temp[k++] = arr[j++];
	}
	while (i <= mid)//若比较完之后，第一个有序区仍有剩余，则直接复制到t数组中
		temp[k++] = arr[i++];
	while (j <= high)//同上
		temp[k++] = arr[j++];
	for (i = low, k = 0; i <= high; i++, k++)//将排好序的存回arr中low到high这区间
		arr[i] = temp[k];
	//delete[]temp;//释放内存，由于指向的是数组，必须用delete []
}
```



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
            ListNode* top = pri_queue.top();pri_queue.pop();
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
        for(int i=1;
            <m;i++)
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

##### 6.2 46 48 49

给定一个 **没有重复** 数字的序列，返回其所有可能的全排列。

```
输入: [1,2,3]
输出:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]
```

```C++
 vector<vector<int>> permute(vector<int>& nums) 
    {
        vector<vector<int>> result;
        permute(nums, result, 0);
        return result;
    }
    void permute(vector<int> & nums, vector<vector<int>> & result, int i){
        if(i == nums.size())
        {
            result.push_back(nums); 
            return;
        }
        for(int j = i; j<nums.size(); j++){
            if(i!=j) swap(nums[i], nums[j]);
            permute(nums, result, i+1);
            if(i!=j) swap(nums[i], nums[j]);
        }
    }
```

48给定一个 n × n 的二维矩阵表示一个图像。

将图像顺时针旋转 90 度。

说明：

你必须在原地旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要使用另一个矩阵来旋转图像。

```C++
void rotate(vector<vector<int>>& matrix) {
        int n=matrix.size();
        
        //进行对角线变换
         for(int i=0;i<n;i++)
         {
             for(int j=i;j<n;j++)
             {
                 swap(matrix[i][j],matrix[j][i]);
             }
         }

        //沿着竖轴翻转
        for(int i=0;i<n;i++)
        {
            reverse(matrix[i].begin(),matrix[i].end());
        }
    }
```

49给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。

```C++
vector<vector<string>> groupAnagrams(vector<string>& strs) {
        
        vector<vector<string>>ans;
        unordered_map<string,vector<string>>mp;
        for(int i=0;i<strs.size();i++)
        {
            string tmp=strs[i];
            sort(tmp.begin(),tmp.end());
           mp[tmp].push_back(strs[i]);
        }
        for(auto at:mp)
        ans.push_back(at.second);
        return ans;

    }
  
```

##### 6.4 56 75

56给出一个区间的集合，请合并所有重叠的区间。

思路
对 vector<vector<int>> 排序，需要按照先比较区间开始，如果相同再比较区间结束，使用默认的排序规则即可
使用双指针，左边指针指向当前区间的开始
使用一个变量来记录连续的范围 t
右指针开始往后寻找，如果后续的区间的开始值比 t 还小，说明重复了，可以归并到一起
此时更新更大的结束值到 t
直到区间断开，将 t 作为区间结束，存储到答案里
然后移动左指针，跳过中间已经合并的区间



```C++
 vector<vector<int>> merge(vector<vector<int>>& intervals) {
        vector<vector<int>>ans;
        if(intervals.size()==0)
        return ans;
        sort(intervals.begin(),intervals.end());
        
        for(int i=0;i<intervals.size();)
        {
            int start=intervals[i][0];
            int end=intervals[i][1];
            int j=i+1;
            while(j<intervals.size()&&intervals[j][0]<=end)
            {
                end=max(end,intervals[j][1]);
                j++;
            }
            ans.push_back({start,end});
            i=j;
        }
        return ans;
    }
```

75给定一个包含红色、白色和蓝色，一共 n 个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

```
输入: [2,0,2,1,1,0]
输出: [0,0,1,1,2,2]
```

tips：直接快排简单粗暴

```C++
 void sortColors(vector<int>& nums) {
        int i=0;int j=nums.size()-1;
       quicksort(nums,i,j);
    }
    void quicksort(vector<int> &arr,int low,int high)
    {
        if(low<high)
        {
            int i=low,j=high,pivot=arr[i];
            while(i<j)
            {
                while(i<j&&arr[j]>=pivot)
                j--;
                swap(arr[i],arr[j]);
                while(i<j&&arr[i]<=pivot)
                i++;
                swap(arr[i],arr[j]);
            }
            arr[i]=pivot;
            quicksort(arr,low,i-1);
            quicksort(arr,i+1,high);
        }
    }
```

##### 面试题29 78

29顺时针打印矩阵

```C++
 if(matrix.size() == 0 || matrix[0].size() == 0) return {};
      /*设置上下左右四个界限*/
      vector<int> res; /*存储遍历结果*/
      int top = 0;
      int bottom = matrix.size() - 1;
      int left = 0;
      int right = matrix[0].size() - 1;
      /*此算法模拟顺时针输出的过程，请联想打印过程*/
      while(true)
      {
          /*1.top行从左到右遍历*/
          for(int i=left;i<=right;i++){
              res.push_back(matrix[top][i]);
          }
          /*top移动至下一行，并进行边界检测*/
          top++;
          if(top > bottom ) break;

          /*2.right列从上到下遍历*/
          for(int i=top;i<=bottom;i++){
              res.push_back(matrix[i][right]);
          }
          /*right左移，并进行边界检测*/
          right--;
          if(right < left) break;
          
          /*3.bottom行从右往左遍历*/
          for(int i = right;i>=left;i--){
              res.push_back(matrix[bottom][i]);
          }
          /*bottom行上移，并进行边界检测*/
          bottom -- ;
          if(bottom < top) break;
          /*4.left列从下往上遍历*/
          for(int i=bottom;i>=top;i--){
              res.push_back(matrix[i][left]);
          }
          /*left右移，并进行边界检测*/
          left++;
          if(left > right) break;
      }
      /*返回遍历结果*/
      return res;
 

    }
```

78.给定一组**不含重复元素**的整数数组 *nums*，返回该数组所有可能的子集（幂集）。

```C++
解法一：逐层添加
vector<vector<int>> subsets(vector<int>& nums) {
       vector<vector<int>>ans(1);
       for(int i=0;i<nums.size();i++)
       {
           int tmp_size=ans.size();
           for(int j=0;j<tmp_size;j++)
           {
               vector<int>tmp=ans[j];
               tmp.push_back(nums[i]);
               ans.push_back(tmp);
           }
       }
       return ans;
    }
```

```C++
//回溯算法
vector<vector<int>> subsets(vector<int>& nums) {
       vector<vector<int>>ans;
        vector<int>tmp;
        dfs(ans,tmp,nums,0);
       return ans;
    }
    void dfs(vector<vector<int>>&ans,vector<int>tmp,vector<int>& nums,int num)
    {
        if(num<=nums.size())
        {
            ans.push_back(tmp);
        }
        for(int i=num;i<nums.size();i++)
        {
            tmp.push_back(nums[i]);
            dfs(ans,tmp,nums,i+1);
            tmp.pop_back;
        }
    }
```

##### 6.8 104 105

给定一个二叉树，找出其最大深度。二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。

**说明:** 叶子节点是指没有子节点的节点。

```C++
//层序遍历
int maxDepth(TreeNode* root) {
         if(root==NULL)
         return 0;
         int ans=0;
         queue<TreeNode*>que;
         que.push(root);
         while(!que.empty())
         {
             ans+=1;
             int num=que.size();
             for(int i=0;i<num;i++)
             {
                 TreeNode* top=que.front();
                 que.pop();
                 if(top->left)
                 que.push(top->left);
                 if(top->right)
                 que.push(top->right);

             }
         }
         return ans;
    }
```

105

```C++
 TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
          int n = preorder.size();
        int m = inorder.size();
        if(n!=m || n == 0)
            return NULL;
        return construct(preorder, inorder, 0, n-1, 0, m-1);
    }
    TreeNode* construct(vector<int>& pre, vector<int>& vin, int l1, int r1, int l2, int r2)
    {
        TreeNode* root = new TreeNode(pre[l1]);
        if(r1 == l1)
        {
            return root;
        }
        int val = pre[l1];
        int index;
        for(index = l2; index <= r2; index ++)
        {
            if(vin[index] == val)
                break;
        }
        int left_tree_len  = index - l2;
        int right_tree_len = r2 - index;
        if(left_tree_len > 0)
            root->left = construct(pre, vin, l1+1, l1+left_tree_len, l2, index-1);
        if(right_tree_len >0 )
            root->right = construct(pre, vin, l1+1+left_tree_len, r1, index+1, r2);
        return root;
    }
```

##### 6.9 面试46 25

给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。

 

```
输入: 12258
输出: 5
解释: 12258有5种不同的翻译，分别是"bccfi", "bwfi", "bczi", "mcfi"和"mzi"
```

tips：看i前面两个数能组成几种结构，如0X或者26以上只有一种解释方法，对应的解释为aX或数分开解释，在10~25有两种解释 合并算一个或者分开翻译

```C++
int translateNum(int num) {
    string value=to_string(num);
    vector<int> dp(value.size()+1, 0);
    dp[0]=1;dp[1]=1;
    for(int i=2;i<=value.size();i++)
    {
       if (value[i - 2] != '0' && 10 * (value[i - 2] - '0') + value[i - 1] - '0' < 26) 
         dp[i] = dp[i-1]+dp[i - 2];
        else
         dp[i] = dp[i-1];
    }
       return dp[value.size()];
    }
```

```
ListNode* reverseList(ListNode* head, ListNode* tail) {
    ListNode* pPrev = tail;
    ListNode* pCurr = head;
    while (pCurr != tail) {
        ListNode* pNext = pCurr->next;
        pCurr->next = pPrev;
        pPrev = pCurr;
        pCurr = pNext;
    }
    return pPrev;
}

ListNode* reverseKGroup(ListNode* head, int k)
{
    ListNode* dummy = new ListNode(0);
    dummy->next = head;
    ListNode* pPrev = dummy;
    ListNode* pCurr = head;
    while (pCurr != nullptr) {
        int i = 0;
        while (i++ < k && pCurr != nullptr) {
            pCurr = pCurr->next;
        }
        if (i != k + 1) break;

        ListNode* pTemp = pPrev->next;
        pPrev->next = reverseList(pTemp, pCurr);
        pPrev = pTemp;
    }
    return dummy->next;
}

。
```

##### 6.11  79 98

79给定一个二维网格和一个单词，找出该单词是否存在于网格中。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

```
board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

给定 word = "ABCCED", 返回 true
给定 word = "SEE", 返回 true
给定 word = "ABCB", 返回 false
```



```C++
bool exist(vector<vector<char>>& board, string word) {
        if(board.size()==0)
        return false;
        for(int i=0;i<board.size();i++)
        {
            for(int j=0;j<board[0].size();j++)
            {
                if(dfs(board,word,i,j,0))
                return true;
            }
        }
        return false;
    }
   bool dfs(vector<vector<char>>& board, string word,int i,int j,int len)
   {
       if(i>=board.size()||j>=board[0].size()||i<0||j<0||board[i]					  [j]!=word[len]||len>=word.size())
       {
           return false;
       }
       if(board[i][j]==word[len]&&len==word.size()-1)
       {
           return true;
       }
       char tmp=board[i][j];
       board[i][j]='0';
       bool ans=dfs(board,word,i+1,j,len+1)||dfs(board,word,i-1,j,len+1)||dfs(board,word,i,j+1,len+1)||dfs              (board,word,i,j-1,len+1);
        board[i][j]=tmp;
        return ans;
   }
```

98给定一个二叉树，判断其是否是一个有效的二叉搜索树。

假设一个二叉搜索树具有如下特征：

节点的左子树只包含小于当前节点的数。
节点的右子树只包含大于当前节点的数。
所有左子树和右子树自身必须也是二叉搜索树。

tips：二叉搜索树中序遍历是单调递增的，所以先中序后判断即可

```C++
bool isValidBST(TreeNode* root) {
        if(root==NULL)
        return true;
        vector<int>ans;
        inorder(root,ans);
        for(int i=1;i<ans.size();i++)
        {
            if(ans[i]<=ans[i-1])
            return false;
        }
        return true;
        
    }
    void inorder(TreeNode* root,vector<int>&ans)
    {
        
        if(root!=NULL)
        {
            inorder(root->left,ans);
            ans.push_back(root->val);
            inorder(root->right,ans);
        }
    }
```

##### 6.12  739 15

请根据每日 气温 列表，重新生成一个列表。对应位置的输出为：要想观测到更高的气温，至少需要等待的天数。如果气温在这之后都不会升高，请在该位置用 0 来代替。

例如，给定一个列表 temperatures = [73, 74, 75, 71, 69, 72, 76, 73]，你的输出应该是 [1, 1, 4, 2, 1, 1, 0, 0]。

提示：气温 列表长度的范围是 [1, 30000]。每个气温的值的均为华氏度，都是在 [30, 100] 范围内的整数。

```
tips可以维护一个存储下标的单调栈，从栈底到栈顶的下标对应的温度列表中的温度依次递减。如果一个下标在单调栈里，则表示尚未找到下一次温度更高的下标。

正向遍历温度列表。对于温度列表中的每个元素 T[i]，如果栈为空，则直接将 i 进栈，如果栈不为空，则比较栈顶元素 prevIndex 对应的温度 T[prevIndex] 和当前温度 T[i]，如果 T[i] > T[prevIndex]，则将 prevIndex 移除，并将 prevIndex 对应的等待天数赋为 i - prevIndex，重复上述操作直到栈为空或者栈顶元素对应的温度小于等于当前温度，然后将 i 进栈。

为什么可以在弹栈的时候更新 ans[prevIndex] 呢？因为在这种情况下，即将进栈的 i 对应的 T[i] 一定是 T[prevIndex] 右边第一个比它大的元素，试想如果 prevIndex 和 i 有比它大的元素，假设下标为 j，那么 prevIndex 一定会在下标 j 的那一轮被弹掉。

由于单调栈满足从栈底到栈顶元素对应的温度递减，因此每次有元素进栈时，会将温度更低的元素全部移除，并更新出栈元素对应的等待天数，这样可以确保等待天数一定是最小的。

以下用一个具体的例子帮助读者理解单调栈。对于温度列表 [73,74,75,71,69,72,76,73][73,74,75,71,69,72,76,73]，单调栈stack 的初始状态为空，答案 ans 的初始状态是 [0,0,0,0,0,0,0,0][0,0,0,0,0,0,0,0]，按照以下步骤更新单调栈和答案，其中单调栈内的元素都是下标，括号内的数字表示下标在温度列表中对应的温度。

当 i=0i=0 时，单调栈为空，因此将 00 进栈。

stack=[0(73)]

ans=[0,0,0,0,0,0,0,0]

当 i=1i=1 时，由于 7474 大于 7373，因此移除栈顶元素 00，赋值 ans[0]:=1-0ans[0]:=1−0，将 11 进栈。

\textit{stack}=[1(74)]stack=[1(74)]

\textit{ans}=[1,0,0,0,0,0,0,0]ans=[1,0,0,0,0,0,0,0]

当 i=2i=2 时，由于 7575 大于 7474，因此移除栈顶元素 11，赋值 ans[1]:=2-1ans[1]:=2−1，将 22 进栈。

\textit{stack}=[2(75)]stack=[2(75)]

\textit{ans}=[1,1,0,0,0,0,0,0]ans=[1,1,0,0,0,0,0,0]

当 i=3i=3 时，由于 7171 小于 7575，因此将 33 进栈。

\textit{stack}=[2(75),3(71)]stack=[2(75),3(71)]

\textit{ans}=[1,1,0,0,0,0,0,0]ans=[1,1,0,0,0,0,0,0]

当 i=4i=4 时，由于 6969 小于 7171，因此将 44 进栈。

\textit{stack}=[2(75),3(71),4(69)]stack=[2(75),3(71),4(69)]

\textit{ans}=[1,1,0,0,0,0,0,0]ans=[1,1,0,0,0,0,0,0]

当 i=5i=5 时，由于 7272 大于 6969 和 7171，因此依次移除栈顶元素 44 和 33，赋值 ans[4]:=5-4ans[4]:=5−4 和 ans[3]:=5-3ans[3]:=5−3，将 55 进栈。

\textit{stack}=[2(75),5(72)]stack=[2(75),5(72)]

\textit{ans}=[1,1,0,2,1,0,0,0]ans=[1,1,0,2,1,0,0,0]

当 i=6i=6 时，由于 7676 大于 7272 和 7575，因此依次移除栈顶元素 55 和 22，赋值 ans[5]:=6-5ans[5]:=6−5 和 ans[2]:=6-2ans[2]:=6−2，将 66 进栈。

\textit{stack}=[6(76)]stack=[6(76)]

\textit{ans}=[1,1,4,2,1,1,0,0]ans=[1,1,4,2,1,1,0,0]

当 i=7i=7 时，由于 7373 小于 7676，因此将 77 进栈。

\textit{stack}=[6(76),7(73)]stack=[6(76),7(73)]

\textit{ans}=[1,1,4,2,1,1,0,0]ans=[1,1,4,2,1,1,0,0]

JavaPython3C++Golang


作者：LeetCode-Solution
链接：https://leetcode-cn.com/problems/daily-temperatures/solution/mei-ri-wen-du-by-leetcode-solution/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```



```C++
 vector<int> dailyTemperatures(vector<int>& T) {
        int num=T.size();
         vector<int>ans(num,0);
        stack<int>stk;
        for(int i=0;i<num;i++)
        {
            while(!stk.empty()&&T[i]>T[stk.top()])
            {
                int pre=stk.top();
                ans[pre]=i-pre;
                stk.pop();
            }
            stk.push(i);
        }
        return ans;
    }
```

15 给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有满足条件且不重复的三元组。

注意：答案中不可以包含重复的三元组。



```C++
vector<vector<int>> threeSum(vector<int>& nums) 
    {
        sort(nums.begin(),nums.end());
        vector<vector<int>> ret;
        int n = nums.size(),start = 0,l,r;
        while(start<n){
            if(nums[start]>0)break;
            if(start&&nums[start]==nums[start-1]){start++;continue;}
            l=start+1;
            r = n-1;
            while(l<r){
                int t =nums[start]+nums[l]+nums[r];
                if(!t){
                    ret.push_back({nums[start],nums[l],nums[r]});
                    while (l<r && nums[l] == nums[l+1]) l++; // 去重
                    while (l<r && nums[r] == nums[r-1]) r--; // 去重
                    r--;l++;
                }
                else if(t>0){r--;}
                else if(t<0){l++;}
            }
            start++;
        }
        return ret;
    }
```

