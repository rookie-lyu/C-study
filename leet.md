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
            tmp.pop_back();
        }
    }
```

