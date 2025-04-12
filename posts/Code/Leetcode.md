# 1. Array and String

## Two pointers

> Start the pointers at the edges of the input. Move them towards each other until they meet.

```python
function fn(arr):
    left = 0
    right = arr.length - 1

    while left < right:
        Do some logic here depending on the problem
        Do some more logic here to decide on one of the following:
            1. left++
            2. right--
            3. Both left++ and right--
```
The strength of this technique is that we will never have more than  $O(n)$ iterations for the while loop because the pointers start $n$ away from each other and move at least one step closer in every iteration. Therefore, if we can keep the work inside each iteration at $O(1)$, this technique will result in a linear runtime, which is usually the best possible runtime. 

#### sample Questions
- Given a string `s`, return `true` if it is a `palindrome`, `false` otherwise
- Given a sorted array of unique integers and a target integer, return `true` if there exists a pair of numbers that sum to target, `false` otherwise.
- [Squares of a Sorted Array](https://leetcode.com/problems/squares-of-a-sorted-array/description/)
- [Reverse-words-in-a-string-iii](https://leetcode.com/problems/reverse-words-in-a-string-iii/description/)
- [minimum-consecutive-cards-to-pick-up](https://leetcode.com/problems/minimum-consecutive-cards-to-pick-up/description/)


> Move along both inputs simultaneously until all elements have been checked.

```python
function fn(arr1, arr2):
    i = j = 0
    while i < arr1.length AND j < arr2.length:
        Do some logic here depending on the problem
        Do some more logic here to decide on one of the following:
            1. i++
            2. j++
            3. Both i++ and j++

    // Step 4: make sure both iterables are exhausted
    // Note that only one of these loops would run
    while i < arr1.length:
        Do some logic here depending on the problem
        i++

    while j < arr2.length:
        Do some logic here depending on the problem
        j++
```
#### Sample Questions:
- Given two sorted integer arrays arr1 and arr2, return a new array that combines both of them and is also sorted.

Similar to the first method we looked at, this method will have a linear time complexity of $O(n+m)$ if the work inside the while loop is 
$O(1)$, where $n = arr1.length$ and $m = arr2.length$. This is because at every iteration, we move at least one pointer forward, and the pointers cannot be moved forward more than $n + m$ times without the arrays being exhausted. Let's look at some examples.

## Sliding Window
#### When should we use sliding window?
There is a very common group of problems involving subarrays that can be solved efficiently with sliding window. Let's talk about how to identify these problems.

**First**, the problem will either explicitly or implicitly define criteria that make a subarray "valid". There are 2 components regarding what makes a subarray valid:

A constraint metric. This is some attribute of a subarray. It could be the sum, the number of unique elements, the frequency of a specific element, or any other attribute.
A numeric restriction on the constraint metric. This is what the constraint metric should be for a subarray to be considered valid.
For example, let's say a problem declares a subarray is valid if it has a sum less than or equal to 10. The constraint metric here is the sum of the subarray, and the numeric restriction is <= 10. A subarray is considered valid if its constraint metric conforms to the numeric restriction, i.e. the sum is less than or equal to 10.

**Second**, the problem will ask you to find valid subarrays in some way.

The most common task you will see is finding the best valid subarray. The problem will define what makes a subarray better than another. For example, a problem might ask you to find the longest valid subarray.

Another common task is finding the number of valid subarrays. We will take a look at this later in the article.

> Whenever a problem description talks about subarrays, you should figure out if sliding window is a good option by analyzing the problem description. If you can find the things mentioned above, then it's a good bet.

#### Sample Questions:
- Find the longest subarray with a sum less than or equal to $k$
- Find the longest substring that has at most one "0"
- Find the number of subarrays that have a product less than $k$
- [Max Consecutive Ones III](https://leetcode.com/problems/max-consecutive-ones-iii/description/)


```python
function fn(arr):
    left = 0
    for (int right = 0; right < arr.length; right++):
        Do some logic to "add" element at arr[right] to window

        while WINDOW_IS_INVALID:
            Do some logic to "remove" element at arr[left] from window
            left++

        Do some logic to update the answer
```

sample solution to sum of array = k:

```python
function fn(nums, k):
    left = 0
    curr = 0
    answer = 0
    for (int right = 0; right < nums.length; right++):
        curr += nums[right]
        while (curr > k):
            curr -= nums[left]
            left++

        answer = max(answer, right - left + 1)

    return answer
```

#### Complexity
is $O(n)$ with **amortized** $O(1)$ in the inner while loop.

## Number of subarrays
If a problem asks for the number of subarrays that fit some constraint, we can still use sliding window, but we need to use a neat math trick to calculate the number of subarrays.

Let's say that we are using the sliding window algorithm we have learned and currently have a window `(left, right)`. How many valid windows end at index `right`?

There's the current window `(left, right)`, then `(left + 1, right)`, `(left + 2, right)`, and so on until `(right, right)` (only the element at `right`).

You can fix the right bound and then choose any value between `left` and `right` inclusive for the `left` bound. Therefore, the number of valid windows ending at index `right` is equal to the size of the window, which we know is `right - left + 1`.

## Prefix Sum
Prefix sum is a technique that can be used on arrays (of numbers). The idea is to create an array prefix where `prefix[i]` is the sum of all elements up to the index `i` (inclusive). For example, given `nums = [5, 2, 1, 6, 3, 8]`, we would have `prefix = [5, 7, 8, 14, 17, 25]`.

When a subarray starts at index `0`, it is considered a "prefix" of the array. A prefix sum represents the sum of all prefixes.

Prefix sums allow us to find the sum of any subarray in `O(1)`. If we want the sum of the subarray from `i` to `j` (inclusive), then the answer is `prefix[j] - prefix[i - 1]`, or `prefix[j] - prefix[i] + nums[i]` if you don't want to deal with the out of bounds case when `i = 0`.

This works because `prefix[i - 1]` is the sum of all elements before index `i`. When you subtract this from the sum of all elements up to index `j`, you are left with the sum of all elements starting at index `i` and ending at index `j`, which is exactly what we are looking for.

# 2. Hashing:
Hashing can be used for existance or for counting. Since existance is easy we jump into counting:
### Counting with Hash
Counting is a very common pattern with hash maps. By "counting", we are referring to tracking the frequency of things. This means our hash map will be mapping keys to integers. Anytime you need to count anything, think about using a hash map to do it.

Recall that when we were looking at **sliding windows**, some problems had their constraint as limiting the amount of a certain element in the window. For example, longest substring with at most `k` `0`s. In those problems, we could simply use an integer variable curr because we are only focused on **one element** (we only cared about `0`). A hash map opens the door to solving problems where the constraint involves **multiple elements**. Let's start by looking at a sliding window example that leverages a hash map.


## Count the number of subarrays with an "exact" constraint
In the sliding window article from previous section, we talked about a pattern "find the number of subarrays/substrings that fit a constraint". In those problems, if you had a window between left and right that fit the constraint, then all windows from `x` to `right` also fit the constraint, where `left < x <= right`.

For this pattern, we will be looking at problems with stricter constraints, so that the property just mentioned is not necessarily true.

For example, **"Find the number of subarrays that have a sum less than k"** with an input that only has positive numbers would be solved with sliding window. In this section, we would be talking about questions like **"Find the number of subarrays that have a sum exactly equal to k"**.

At first, some of these problems seem very difficult. However, the pattern is very simple once you learn it, and you'll see how similar the code is for each problem that falls in this pattern. To understand the algorithm, we need to recall the concept of **prefix sums**.

With a prefix sum, you can find the sum of subarrays by taking the difference between two prefix sums. Let's say that you wanted to find subarrays that had a sum exactly equal to k, and you also had a prefix sum of the input. You know that any difference in the prefix sum equal to k represents a subarray with a sum equal to k. So how do we find these differences?

Let's first declare a hash map counts that maps prefix sums to how often they occur (a number could appear multiple times in a prefix sum if the input has negative numbers, for example, given `nums = [1, -1, 1]`, the prefix sum is `[1, 0, 1]` and `1` appears twice). We need to initialize `counts[0] = 1`. This is because the empty `prefix []` has a sum of `0`. You'll see why this is necessary in a second.

Next, let's declare our `answer` variable and `curr`. As we iterate over the input, `curr` will represent the sum of all elements we have iterated over so far (the sum of the prefix up to the current element).

Now, we iterate over the input. At each element, we update `curr` and also maintain counts by incrementing the frequency of curr by `1` 
- remember, `counts` is counting how many times we have seen each prefix sum. Before we update counts however, we first need to update the `answer`.

How do we update the answer? Recall that in the sliding window article, when we were looking for the "number of subarrays", we focused on each index and figured out how many valid subarrays ended at the current index. We will do the same thing here. Let's say that we're at an index `i`. We know a few things:

- Up until this point, `curr` stores the prefix of all elements up to `i`.
- We have stored all other prefixes before `i` inside of `counts`.
The difference between any two prefix sums represents a subarray. For example, if you wanted the subarray starting at index `3` and ending at index `8`, you would take the prefix up to `8` and subtract the prefix up to `2` from it.
Now, imagine there exists a subarray ending at `i` with a sum of `k`. We don't know where this subarray starts, we just know it exists - let's say it starts at index `j`. What would the prefix sum be ending at `j - 1`? Well, according to our assumptions, the sum of the subarray from `j` to `i` is `k` and the sum of the prefix up to `i` is `curr`. Thus, you can find the prefix sum ending at `j - 1` by subtracting these two: it's `curr - k`.

This is the key idea: if we saw the prefix sum `curr - k` before, it necessarily implies that there is a subarray ending at `i` with a sum of `k`. Again, we don't know where the beginning of this subarray is; we just know it exists, but that's enough to solve the problem.

Therefore, we can increment our answer by `counts[curr - k]`. If the prefix `curr - k` occurred multiple times before (due to negative numbers), then each of those prefixes could be used as a starting point to form a subarray ending at the current index with a sum of `k`. That's why we need to track the frequency.
> Let's use a concrete example to better illustrate this idea. Imagine we had `nums = [0, 1, 2, 3, 4]` and `k = 5`. Let's jump to `i = 3`.
> 
> Currently, `curr = 6` (remember, `curr` is tracking the prefix sum up to `i`). We also have `0`, `1`, and `3` in `counts` (all the prefix sums we have encountered so far).
> 
> At this point, we can see that there is a subarray ending at `i` with a sum of `k`
> - it's `[2, 3]`. How does our algorithm see it though?
>
> The current prefix sum is `6`. We want a subarray with a sum of `5`. Thus, if there was a prefix sum of `1` earlier, you could just subtract that prefix from the current one, and you'll get a subarray sum of `5`. In > > this case, we had a `prefix [0, 1]` which has a prefix sum of `1`. We can subtract that from the current prefix `[0, 1, 2, 3]` and we're left with `[2, 3]`, which has our target sum.
>
> Here's another mathematical way to look at it: we have `curr` and we need to subtract `x` from it to find `k`. The equation is `curr - x = k`. We can rearrange for `x` to get `x = curr - k`.
>

> Given an integer array nums and an integer `k`, find the number of subarrays whose sum is equal to `k`.

> [!Note]
```python
from collections import defaultdict
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        counter = defaultdict(int)
        prefix = ans = 0
        counter[0] = 1
        for item in nums:
            prefix+=item
            ans +=counter[prefix - k]
            counter[prefix]+=1
        
        return ans
```
### Sample Questions:
- [Contiguous-Array](https://leetcode.com/problems/contiguous-array/)
- [Group-Anagrams](https://leetcode.com/problems/group-anagrams/)
- [max-sum-of-a-pair-with-equal-sum-of-digits](https://leetcode.com/problems/max-sum-of-a-pair-with-equal-sum-of-digits/)

## Tricks:
- To check if string `s1` is a permutation of string `s2` you have to check if `Counter(s1) == Counter(s2)`
- Given two strings s and t, determine if they are isomorphic.

Two strings `s` and `t` are isomorphic if the characters in s can be replaced to get `t`.
All occurrences of a character must be replaced with another character while preserving the order of characters. No two characters may map to the same character, but a character may map to itself.

 
Example :
Input: s = "egg", t = "add"
Output: true
Explanation:
The strings s and t can be made identical by:

Mapping 'e' to 'a'.
Mapping 'g' to 'd'.

```python
  class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        if len(set(s))==len(set(t))==len(set(zip(s,t))):
            return True
        else:
            return False    
 '''

# 3. Trees

```python
class TreeNode:
    def __init__(self, val, left, right):
        self.val = val
        self.left = left
        self.right = right
```
## Depth-first search (DFS)
In a DFS, we prioritize depth by traversing as far down the tree as possible in one direction (until reaching a leaf node) before considering the other direction. For example, let's say we choose left as our priority direction. We move exclusively with `node.left` until the left subtree has been fully explored. Then, we move up one step and explore the right subtree ( the right leaf will be the parent we we do DFS on that again) when done again we move up one level and do everything again until we finish traversing all the nodes.

<img src="../../images/DFS.gif"  width="500" height="500">
