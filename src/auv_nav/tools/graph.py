""" A Python Class
A simple Python graph class, demonstrating the essential
facts and functionalities of graphs.
"""


class Graph(object):
    def __init__(self, graph_dict=None):
        """initializes a graph object
        If no dictionary or None is given, an empty dictionary will be used
        """
        if graph_dict is None:
            graph_dict = {}
        self.__graph_dict = graph_dict

    def vertices(self):
        """returns the vertices of a graph"""
        return list(self.__graph_dict.keys())

    def edges(self):
        """returns the edges of a graph"""
        return self.__generate_edges()

    def add_vertex(self, vertex):
        """If the vertex "vertex" is not in
        self.__graph_dict, a key "vertex" with an empty
        list as a value is added to the dictionary.
        Otherwise nothing has to be done.
        """
        if vertex not in self.__graph_dict:
            self.__graph_dict[vertex] = []

    def add_edge(self, edge):
        """assumes that edge is of type set, tuple or list;
        between two vertices can be multiple edges!
        """
        edge = set(edge)
        vertex1 = edge.pop()
        if edge:
            # not a loop
            vertex2 = edge.pop()
        else:
            # a loop
            vertex2 = vertex1
        if vertex1 in self.__graph_dict:
            self.__graph_dict[vertex1].append(vertex2)
        else:
            self.__graph_dict[vertex1] = [vertex2]

    def __generate_edges(self):
        """A static method generating the edges of the
        graph "graph". Edges are represented as sets
        with one (a loop back to the vertex) or two
        vertices
        """
        edges = []
        for vertex in self.__graph_dict:
            for neighbour in self.__graph_dict[vertex]:
                if {neighbour, vertex} not in edges:
                    edges.append({vertex, neighbour})
        return edges

    def __str__(self):
        res = "vertices: "
        for k in self.__graph_dict:
            res += str(k) + " "
        res += "\nedges: "
        for edge in self.__generate_edges():
            res += str(edge) + " "
        return res

    def find_path(self, start_vertex, end_vertex, path=[]):
        """find a path from start_vertex to end_vertex
        in graph"""
        graph = self.__graph_dict
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return path
        if start_vertex not in graph:
            return None
        for vertex in graph[start_vertex]:
            if vertex not in path:
                extended_path = self.find_path(vertex, end_vertex, path)
                if extended_path:
                    return extended_path
        return None

    # def find_all_paths(self, start_vertex, end_vertex, path=[]):
    #     """ find all paths from start_vertex to
    #         end_vertex in graph """
    #     graph = self.__graph_dict
    #     path = path + [start_vertex]
    #     if start_vertex == end_vertex:
    #         return [path]
    #     if start_vertex not in graph:
    #         return []
    #     paths = []
    #     for vertex in graph[start_vertex]:
    #         if vertex not in path:
    #             extended_paths = self.find_all_paths(vertex,
    #                                                  end_vertex,
    #                                                  path)
    #             for p in extended_paths:
    #                 paths.append(p)
    #     return paths

    # def is_connected(self,
    #                  vertices_encountered=None,
    #                  start_vertex=None):
    #     """ determines if the graph is connected """
    #     if vertices_encountered is None:
    #         vertices_encountered = set()
    #     gdict = self.__graph_dict
    #     vertices = list(gdict.keys())  # "list" necessary in Python 3
    #     if not start_vertex:
    #         # chosse a vertex from graph as a starting point
    #         start_vertex = vertices[0]
    #     vertices_encountered.add(start_vertex)
    #     if len(vertices_encountered) != len(vertices):
    #         for vertex in gdict[start_vertex]:
    #             if vertex not in vertices_encountered:
    #                 if self.is_connected(vertices_encountered, vertex):
    #                     return True
    #     else:
    #         return True
    #     return False


if __name__ == "__main__":

    g = {
        "origin": ["cameras", "dvl", "imu", "depth", "altitude"],
        "cameras": ["LM165", "Xviii3", "Xviii5"],
        "dvl": [],
        "imu": [],
        "depth": [],
        "altitude": [],
        "LM165": [],
        "Xviii3": [],
        "Xviii5": [],
    }

    graph = Graph(g)

    graph.add_vertex("chemical")
    graph.add_vertex("chemical")
    graph.add_edge(("origin", "chemical"))

    print(graph)

    print("List of isolated vertices:")
    print(graph.find_isolated_vertices())

    print("""A path from "LM165" to "Xviii3":""")
    print(graph.find_path("LM165", "Xviii3"))

    print("""All paths from "origin" to "Xviii5":""")
    print(graph.find_all_paths("origin", "Xviii5"))

    print("Add vertex 'chemical':")
    graph.add_vertex("chemical")
    print(graph)

    print("Add edge ('origin','chemical'): ")
    graph.add_edge(("origin", "chemical"))
    print(graph)
