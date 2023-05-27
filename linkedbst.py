"""
File: linkedbst.py
Author: Ken Lambert
"""

from math import log
from abstractcollection import AbstractCollection
from bstnode import BSTNode
from linkedstack import LinkedStack
from linkedqueue import LinkedQueue


class LinkedBST(AbstractCollection):
    """An link-based binary search tree implementation."""

    def __init__(self, sourceCollection=None):
        """Sets the initial state of self, which includes the
        contents of sourceCollection, if it's present."""
        self._root = None
        AbstractCollection.__init__(self, sourceCollection)

    # Accessor methods
    def __str__(self):
        """Returns a string representation with the tree rotated
        90 degrees counterclockwise."""

        def recurse(node, level):
            sentence = ""
            if node is not None:
                sentence += recurse(node.right, level + 1)
                sentence += "| " * level
                sentence += str(node.data) + "\n"
                sentence += recurse(node.left, level + 1)
            return sentence

        return recurse(self._root, 0)

    def __iter__(self):
        """Supports a preorder traversal on a view of self."""
        if not self.isEmpty():
            stack = LinkedStack()
            stack.push(self._root)
            while not stack.isEmpty():
                node = stack.pop()
                yield node.data
                if node.right is not None:
                    stack.push(node.right)
                if node.left is not None:
                    stack.push(node.left)

    def preorder(self):
        """Supports a preorder traversal on a view of self."""
        return None

    def inorder(self):
        """Supports an inorder traversal on a view of self."""
        lyst = list()
        stack = []
        current = self._root

        while current or stack:
            while current:
                stack.append(current)
                current = current.left

            current = stack.pop()
            lyst.append(current.data)
            current = current.right

        return iter(lyst)

    def postorder(self):
        """Supports a postorder traversal on a view of self."""
        return None

    def levelorder(self):
        """Supports a levelorder traversal on a view of self."""
        lst = []
        if not self.isEmpty():
            queue = LinkedQueue()
            queue.add(self._root)
            while not queue.isEmpty():
                node = queue.pop()
                lst += [node.data]
                # yield node.data

                if node.right is not None:
                    queue.add(node.right)

        return lst

    def __contains__(self, item):
        """Returns True if target is found or False otherwise."""
        return self.find(item) is not None

    def find(self, item):
        """If item matches an item in self, returns the
        matched item, or None otherwise."""

        current_node = self._root

        while current_node is not None:
            if item == current_node.data:
                return current_node.data
            elif item < current_node.data:
                current_node = current_node.left
            else:
                current_node = current_node.right
        return None

    # Mutator methods

    def clear(self):
        """Makes self become empty."""
        self._root = None
        self._size = 0

    def add(self, item):
        """Adds item to the tree."""

        if self.isEmpty():
            self._root = BSTNode(item)
            self._size += 1
            return

        current_node = self._root

        while True:
            if item < current_node.data:
                if current_node.left is None:
                    current_node.left = BSTNode(item)
                    self._size += 1
                    break
                else:
                    current_node = current_node.left
            else:
                if current_node.right is None:
                    current_node.right = BSTNode(item)
                    self._size += 1
                    break
                else:
                    current_node = current_node.right

    def remove(self, item):
        """Precondition: item is in self.
        Raises: KeyError if item is not in self.
        postcondition: item is removed from self."""
        if not item in self:
            raise KeyError("Item not in tree.""")

        # Helper function to adjust placement of an item
        def lift_max_in_left_subtree_to_top(top):
            # Replace top's datum with the maximum datum in the left subtree
            # Pre:  top has a left child
            # Post: the maximum node in top's left subtree
            #       has been removed
            # Post: top.data = maximum value in top's left subtree
            parent = top
            current_node = top.left
            while not current_node.right is None:
                parent = current_node
                current_node = current_node.right
            top.data = current_node.data
            if parent == top:
                top.left = current_node.left
            else:
                parent.right = current_node.left

        # Begin main part of the method
        if self.isEmpty():
            return None

        # Attempt to locate the node containing the item
        item_removed = None
        pre_root = BSTNode(None)
        pre_root.left = self._root
        parent = pre_root
        direction = 'L'
        current_node = self._root
        while not current_node is None:
            if current_node.data == item:
                item_removed = current_node.data
                break
            parent = current_node
            if current_node.data > item:
                direction = 'L'
                current_node = current_node.left
            else:
                direction = 'R'
                current_node = current_node.right

        # Return None if the item is absent
        if item_removed is None:
            return None

        # The item is present, so remove its node

        # Case 1: The node has a left and a right child
        #         Replace the node's value with the maximum value in the
        #         left subtree
        #         Delete the maximium node in the left subtree
        if not current_node.left is None \
                and not current_node.right is None:
            lift_max_in_left_subtree_to_top(current_node)
        else:

            # Case 2: The node has no left child
            if current_node.left is None:
                new_child = current_node.right

                # Case 3: The node has no right child
            else:
                new_child = current_node.left

                # Case 2 & 3: Tie the parent to the new child
            if direction == 'L':
                parent.left = new_child
            else:
                parent.right = new_child

        # All cases: Reset the root (if it hasn't changed no harm done)
        #            Decrement the collection's size counter
        #            Return the item
        self._size -= 1
        if self.isEmpty():
            self._root = None
        else:
            self._root = pre_root.left
        return item_removed

    def replace(self, item, new_item):
        """
        If item is in self, replaces it with newItem and
        returns the old item, or returns None otherwise."""
        probe = self._root
        while probe is not None:
            if probe.data == item:
                old_data = probe.data
                probe.data = new_item
                return old_data
            if probe.data > item:
                probe = probe.left
            else:
                probe = probe.right
        return None

    def children(self, node):
        '''children finding'''
        if node.left is None and node.right is None:
            return []
        elif node.left is None and node.right:
            return [node.right]
        elif node.left and node.right is None:
            return [node.left]
        return [node.left, node.right]

    def height(self):
        '''
        Return the height of tree
        :return: int
        '''
        if self.isEmpty():
            return -1

        def height1(top):
            '''
            Helper function
            :param top:
            :return:
            '''
            if top.left is None and top.right is None:
                return 0
            return 1+max(height1(c) for c in self.children(top))
        return height1(self._root)

    def is_balanced(self):
        '''
        Return True if tree is balanced
        :return:
        '''
        if 2*log(self._size+1)-1 > self.height():
            return True
        return False

    def range_find(self, low, high):
        '''
        Returns a list of the items in the tree, where low <= item <= high."""
        :param low:
        :param high:
        :return:
        '''
        lst = [elem for elem in self.inorder()]
        return lst[lst.index(low):lst.index(high)+1]

    def rebalance(self):
        '''
        Rebalances the tree.
        :return:
        '''
        lst = [elem for elem in self.inorder()]
        for elem in self.inorder():
            self.remove(elem)

        stack = [lst]
        while stack:
            given = stack.pop()
            if len(given) == 1:
                self.add(given[0])
            elif len(given) > 1:
                root = given[len(given) // 2]
                self.add(root)

                lll = len(given) // 2
                given_pre = given[:lll]
                given_after = given[lll+1:]

                stack.append(given_after)
                stack.append(given_pre)

    def successor(self, item):
        """
        Returns the smallest item that is larger than
        item, or None if there is no such item.
        :param item:
        :type item:
        :return:
        :rtype:
        """
        min_max = None
        probe = self._root
        while True:
            if item < probe.data:

                min_max = probe.data
                if probe.left:
                    probe = probe.left
                else:
                    return min_max
            elif item > probe.data:
                if probe.right:
                    probe = probe.right
                else:
                    return min_max
            elif item == probe.data and probe.right:
                return probe.right.data
            else:
                return min_max

    def predecessor(self, item):
        """
        Returns the largest item that is smaller than
        item, or None if there is no such item.
        :param item:
        :type item:
        :return:
        :rtype:
        """
        max_min = None
        probe = self._root
        while True:
            if item < probe.data:
                if probe.left:
                    probe = probe.left
                else:
                    return max_min
            elif item > probe.data:
                max_min = probe.data
                if probe.right:
                    probe = probe.right
                else:
                    return max_min
            elif item == probe.data and probe.left:
                return probe.left.data
            else:
                return max_min

    def demo_bst(self, path):
        """
        Demonstration of efficiency binary search tree for the search tasks.
        :param path:
        :type path:
        :return:
        :rtype:
        """
        import random
        import time

        with open(path, 'r') as file:
            dictionary = [word.strip() for word in file.readlines()]
        sorted_dictionary = sorted(dictionary)
        random_words = random.choices(dictionary, k=10000)

        # sorted list
        start_time = time.time()
        for word in random_words:
            _ = word in sorted_dictionary
        list_search_time = time.time() - start_time

        print(
            f"Time to search 10,000 random words in a sorted list: {list_search_time:.2f} seconds")

        # in the sorted BST
        bst = LinkedBST()
        for elem in sorted_dictionary:
            bst.add(elem)
        start_time = time.time()
        for word in random_words:
            bst.find(word)
        bst_sorted_search_time = time.time() - start_time

        print(
            f"Time to search 10,000 random words in the sorted BST: {bst_sorted_search_time:.2f} seconds")

        in the BST
        bst2 = LinkedBST()
        for elem in dictionary:
            bst2.add(elem)
            # print('done')
        start_time = time.time()
        for word in random_words:
            bst2.find(word)
            # print('done')
        bst_search_time = time.time() - start_time

        print(
            f"Time to search 10,000 random words in the BST: {bst_search_time:.2f} seconds")

        # in the balanced BST
        bst3 = LinkedBST()
        for elem in dictionary:
            bst3.add(elem)
        bst3.rebalance()
        start_time = time.time()
        for word in random_words:
            bst3.find(word)
        rebalanced_bst_search_time = time.time() - start_time

        print(
            f"Time to search 10,000 random words in the BST: {rebalanced_bst_search_time:.2f} seconds")


# # example
# owrds = LinkedBST()
# owrds.demo_bst('/Users/dyshleva/Desktop/current/2term/1705/1task/words.txt')
