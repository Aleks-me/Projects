import csv
from collections import deque
import itertools


class DAG:
    """Directed acyclic graph for multiple branching pipeline"""

    def __init__(self):
        self.graph = {}

    def in_degrees(self):
        """How many connections to node there are."""

        degrees = {}
        for node in self.graph:
            if node not in degrees:
                degrees[node] = 0
            for pointed in self.graph[node]:
                if pointed not in degrees:
                    degrees[pointed] = 0
                degrees[pointed] += 1
        return degrees

    def sort(self):
        """Sorting tasks by num of connections."""

        in_degrees = self.in_degrees()
        to_visit = deque()
        for node in self.graph:
            if in_degrees[node] == 0:
                to_visit.append(node)

        searched = []
        while to_visit:
            node = to_visit.popleft()
            for pointer in self.graph[node]:
                in_degrees[pointer] -= 1
                if in_degrees[pointer] == 0:
                    to_visit.append(pointer)
            searched.append(node)
        return searched

    def add(self, node, bound_to=None):
        """Add tasks to queue."""

        # For newly created task:
        if node not in self.graph:
            self.graph[node] = []
        # Bound task to another task:
        if bound_to:
            if bound_to not in self.graph:
                self.graph[bound_to] = []
            self.graph[node].append(bound_to)
        # If there's a cycle:
        if len(self.sort()) != len(self.graph):
            print("Cycle in graph detected!")
            raise Exception


class Pipeline:
    """Task collection."""

    def __init__(self):
        self.tasks = DAG()

    def task(self, depends_on=None):
        """Decorate function with pipeline.task()"""

        def inner(function):
            self.tasks.add(function)
            if depends_on:
                self.tasks.add(depends_on, function)
            return function
        return inner

    def run(self):
        """Run queue of functions."""

        scheduled = self.tasks.sort()
        completed = {}

        for task in scheduled:
            for node, values in self.tasks.graph.items():
                if task in values:
                    completed[task] = task(completed[node])
            if task not in completed:
                completed[task] = task()
        return completed


def build_csv(lines, header=None, file=None):
    if header:
        lines = itertools.chain([header], lines)
    writer = csv.writer(file, delimiter=',')
    writer.writerows(lines)
    file.seek(0)
    return file
