import threading
import queue
import concurrent.futures

import time
from datetime import datetime, timedelta

# class DagThreader:
#     def __init__(self, io_function):
#         self.io_function = io_function

#     def worker_function(self, item):
#         try:
#             result = self.io_function(item[1])
#             return (item[0], result)
#         except Exception as e:
#             print(f"Error processing item {item[0]}: {e}")

#     def run_threader(self, input_list, max_size, num_workers):
#         with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
#             items_with_index = list(enumerate(input_list))
#             results = list(executor.map(self.worker_function, items_with_index))
        
#         final_results = [result[1] for result in sorted(results, key=lambda item: item[0])]
#         return final_results


class DagThreader:
    """
    A class for efficiently processing a list of items using multithreading.

    Args:
        io_function (callable): A function that takes input and produces output.

    Attributes:
        io_function (callable): The input-output function.
    """

    def __init__(self, io_function):
        """
        Initialize a DagThreader instance.

        Args:
            io_function (callable): A function that takes input and produces output.
        """
        self.io_function = io_function
        self.result_queue = queue.Queue()

    def worker_function(self, total_count, input_item):
        """
        A worker function that applies the io_function to input and stores the result.

        Args:
            total_count (int): The total count of items processed.
            input_item: The input item to be processed.
        """
        try:
            result = self.io_function(input_item)
            self.result_queue.put((total_count, result))
        except Exception as e:
            print(f"Error processing item {total_count}: {e}")

    def run_threader(self, input_list, max_size):
        """
        Execute the multithreaded processing on the input list.

        Args:
            input_list (list): The list of items to be processed.
            max_size (int): The maximum size of each chunk for multithreading.

        Returns:
            list: A list of processed results.
        """
        results = []  # Store results in a list
        splits = self.split_list(input_list, max_size)  # Split input data into manageable chunks
        total_count = 0

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for this_split in splits:  # Iterate over splits
                futures = []

                for worker_input in this_split:
                    future = executor.submit(self.worker_function, total_count, worker_input)
                    futures.append(future)
                    total_count += 1

                # Wait for all futures in the split to complete
                concurrent.futures.wait(futures)

                # Retrieve results from the queue
                while not self.result_queue.empty():
                    result = self.result_queue.get()
                    results.append(result)

        # Sort results based on thread IDs
        sorted_results = sorted(results, key=lambda item: item[0])

        # Extract only the results (discard thread IDs)
        final_results = [item[1] for item in sorted_results]

        return final_results

    @staticmethod
    def split_list(input_list, max_size):
        """
        Split a list into smaller chunks for multithreading.

        Args:
            input_list (list): The list to be split.
            max_size (int): The maximum size of each chunk.

        Returns:
            list: A list of smaller chunks of the input list.
        """
        splits = [input_list[i:i + max_size] for i in range(0, len(input_list), max_size)]
        return splits


# ***************************************************************************************************************************************************
# ***************************************************************************************************************************************************
# ORIGINAL
# ***************************************************************************************************************************************************
# import threading
# import queue

# class DagThreader(object):
#     """
#     A class for efficiently processing a list of items using multithreading.

#     Args:
#         io_function (callable): A function that takes input and produces output.

#     Attributes:
#         io_function (callable): The input-output function.
#     """

#     def __init__(self, io_function):
#         """
#         Initialize a DagThreader instance.

#         Args:
#             io_function (callable): A function that takes input and produces output.
#         """
#         self.io_function = io_function
#         self.result_queue = queue.Queue()

#     def worker_function(self, total_count, inp):
#         """
#         A worker function that applies the io_function to input and stores the result.

#         Args:
#             total_count (int): The total count of items processed.
#             inp: The input item to be processed.
#         """
#         result = self.io_function(inp)
#         self.result_queue.put((total_count, result))  # Add result to the result queue

#     def run_threader(self, input_list, max_size):
#         """
#         Execute the multithreaded processing on the input list.

#         Args:
#             input_list (list): The list of items to be processed.
#             max_size (int): The maximum size of each chunk for multithreading.

#         Returns:
#             list: A list of processed results.
#         """
#         results = []  # Store results in a list
#         splits = self.split_list(input_list, max_size)  # Split input data into manageable chunks
#         total_count = 0

#         for this_split in splits:  # Iterate over splits
#             threads = []

#             for thread_id, worker_input in enumerate(this_split):
#                 thread = threading.Thread(
#                     target=self.worker_function, args=(total_count, worker_input,))
#                 threads.append(thread)
#                 thread.start()
#                 total_count += 1

#             # Wait for all threads in the split to complete
#             for thread in threads:
#                 thread.join()

#             # Sort the list of threads based on thread IDs
#             threads = sorted(threads, key=lambda t: t.ident)

#             # Retrieve results from the queue
#             while not self.result_queue.empty():
#                 result = self.result_queue.get()
#                 results.append(result)

#             # Clear the threads list for the next split
#             threads.clear()

#         # Sort results based on thread IDs
#         sorted_results = sorted(results, key=lambda item: item[0])

#         # Extract only the results (discard thread IDs)
#         final_results = [item[1] for item in sorted_results]

#         return final_results

#     @staticmethod
#     def split_list(input_list, max_size):
#         """
#         Split a list into smaller chunks for multithreading.

#         Args:
#             input_list (list): The list to be split.
#             max_size (int): The maximum size of each chunk.

#         Returns:
#             list: A list of smaller chunks of the input list.
#         """
#         splits = []

#         for i in range(0, len(input_list), max_size):
#             split = input_list[i:i + max_size]
#             splits.append(split)
#         return splits


# ***************************************************************************************************************************************************
# ***************************************************************************************************************************************************
# ***************************************************************************************************************************************************

