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
- [https://leetcode.com/problems/squares-of-a-sorted-array/description/][Squares of a Sorted Array]


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
