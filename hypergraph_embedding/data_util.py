# This file is responsible for preparing experimental data
from . import Hypergraph
from .hypergraph_util import *
from collections import namedtuple
import logging
import json
import requests
from tqdm import tqdm
from scipy.io import mmread
from scipy.io import mmwrite

log = logging.getLogger()

global PARSING_OPTIONS


def ParseRawIntoHypergraph(args, raw_data_path):
  log.info("Parsing %s with %s method", raw_data_path, args.raw_data_format)
  hypergraph = PARSING_OPTIONS[args.raw_data_format](raw_data_path)
  if args.name:
    log.info("Setting hypergraph name to %s", args.name)
    log.info("Good name!")
    hypergraph.name = args.name
  else:
    log.info("Setting hypergraph name to %s", args.hypergraph)
    log.info("Bad name :(")
    hypergraph.name = args.hypergraph
  return hypergraph


# Used to store paper data
Paper = namedtuple("Paper", ['title', 'authors'])


def SnapCommunityToHypergraph(path):
  with open(path, 'r') as snap_source:
    hypergraph = Hypergraph()
    for edge_idx, node_str in enumerate(snap_source):
      for node_idx in node_str.split():
        AddNodeToEdge(hypergraph, int(node_idx), edge_idx)
    return hypergraph


def AMinerToHypergraph(aminer_source):
  return PapersToHypergraph(ParseAMiner(aminer_source))


def ParseAMiner(path):
  """
    Parses data in AMiner's format.
    Ignores all fields except title and authors
    More information on this format here: https://aminer.org/aminernetwork

    Input:
      - aminer_source : a file-like object
    Output: (yield)
      - a list of Papers (named tuple)
   """
  with open(path, 'r') as aminer_source:
    log.info("Parsing AMiner data")
    last_seen_title = None
    for line in aminer_source:
      if line.startswith("#*"):  # paper title line
        last_seen_title = line[2:].strip()
      elif line.startswith("#@"):  # authors line
        authors = line[2:].strip().split(';')
        yield Paper(title=last_seen_title, authors=authors)


def PapersToHypergraph(parser):
  """
    Converts paper data into hypergraph.
    Input:
      - A iterable type that supplies Paper tuples
    Output:
      - A hypergraph
    """
  log.info("Converting papers to hypergraph")
  title2idx = {}
  author2idx = {}
  result = Hypergraph()
  for paper in parser:
    if paper.title not in title2idx:
      title2idx[paper.title] = len(title2idx)
    for author in paper.authors:
      if author not in author2idx:
        author2idx[author] = len(author2idx)
      AddNodeToEdge(result, author2idx[author], title2idx[paper.title], author,
                    paper.title)
  return result


def CleanHypergraph(original_hg, min_degree=2):
  "Iterativly removes nodes / edges with degree smaller than min_degree"
  "Performs operations on copy, does not change original."
  new_hg = Hypergraph()
  new_hg.CopyFrom(original_hg)
  log.info("Removing all nodes / edges with degree < %i", min_degree)
  while len(new_hg.node) and len(new_hg.edge):
    troubled_nodes = [
        node_idx for node_idx, node in new_hg.node.items()
        if len(node.edges) < min_degree
    ]
    for node_idx in troubled_nodes:
      RemoveNode(new_hg, node_idx)
    troubled_edges = [
        edge_idx for edge_idx, edge in new_hg.edge.items()
        if len(edge.nodes) < min_degree
    ]
    for edge_idx in troubled_edges:
      RemoveEdge(new_hg, edge_idx)
    if len(troubled_nodes) == 0 and len(troubled_edges) == 0:
      break
  return new_hg


def DownloadMadGrades(api_token):
  instructor_url = "https://api.madgrades.com/v1/instructors"
  courses_url = "https://api.madgrades.com/v1/courses"

  def get_instructors_id_name(_json):
    return [(r['id'], r['name']) for r in _json['results']]

  def get_courses_uuid_name(_json):
    return [(r['uuid'], r['name']) for r in _json['results']]

  def get_instructors_on_page(page):
    response = requests.get(
        instructor_url,
        headers={"Authorization": "Token token={}".format(api_token)},
        data={'page': page})
    instructor_json = json.loads(response.text)
    total_pages = instructor_json['totalPages']
    instructors_id_name = get_instructors_id_name(instructor_json)
    return instructors_id_name, total_pages

  def get_courses_for_instructor(instructor_id, page):
    response = requests.get(
        courses_url,
        headers={"Authorization": "Token token={}".format(api_token)},
        data={
            'instructor': instructor_id,
            'page': page
        })
    course_json = json.loads(response.text)
    courses_uuid_name = get_courses_uuid_name(course_json)
    total_pages = course_json['totalPages']
    return courses_uuid_name, total_pages

  instructors_id_name, total_pages = get_instructors_on_page(1)
  for page in tqdm(range(2, total_pages + 1)):
    instructors_id_name.extend(get_instructors_on_page(page)[0])

  uuid_map = {}
  result = Hypergraph()
  for instructor_id, instructor_name in tqdm(instructors_id_name):
    result.node[instructor_id].name = instructor_name
    courses_uuid_name, total_pages = get_courses_for_instructor(
        instructor_id, 1)
    for page in range(2, total_pages + 1):
      courses_uuid_name.extend(
          get_courses_for_instructor(instructor_id, page)[0])
    for course_uuid, course_name in courses_uuid_name:
      if course_uuid not in uuid_map:
        uuid_map[course_uuid] = len(uuid_map)
        if course_name is not None:
          result.edge[uuid_map[course_uuid]].name = course_name
      AddNodeToEdge(result, instructor_id, uuid_map[course_uuid])
  return result


def LoadMTX(path):
  mtx = mmread(str(path))
  hypergraph = FromSparseMatrix(mtx.T)
  return hypergraph

def SaveMTX(hypergraph, path):
  mtx_mat = ToEdgeCsrMatrix(hypergraph).astype(np.int32)
  mmwrite(str(path), mtx_mat, comment=hypergraph.name)

def LoadHMetis(path):
  hypergraph = Hypergraph()
  with open(path) as hmetis_file:
    next(hmetis_file)
    for edge_idx, line in enumerate(hmetis_file):
      for node_idx in [int(t)-1 for t in line.strip().split()]:
        AddNodeToEdge(hypergraph, node_idx, edge_idx)
  return hypergraph

def SaveHMetis(hypergraph, path):
  # Read: http://glaros.dtc.umn.edu/gkhome/fetch/sw/hmetis/manual.pdf
  # hmetis requires indices in order
  with open(path, 'w') as hmetis_file:
    # 11 refers to both weighted nodes and hyperedges
    hmetis_file.write("{} {} 11\n".format(len(hypergraph.edge),
                                          len(hypergraph.node)))
    for edge_idx in range(len(hypergraph.edge)):
      hmetis_file.write(str(int(hypergraph.edge[edge_idx].weight)))
      for node_idx in hypergraph.edge[edge_idx].nodes:
        hmetis_file.write(" ")
        # indices are all positive
        hmetis_file.write(str(node_idx + 1))
      hmetis_file.write("\n")
    for node_idx in range(len(hypergraph.node)):
      hmetis_file.write(str(int(hypergraph.node[node_idx].weight)))
      hmetis_file.write("\n")

def SaveEdgeList(hypergraph, data_path, metadata_path, is_weighted=False,
    only_one_side=False):
  hypergraph, node2original, edge2original = CompressRange(hypergraph)
  node2inc = {node_idx: node_idx+1 for node_idx in hypergraph.node}
  max_node_idx = max(hypergraph.node)+1
  edge2inc = {edge_idx: edge_idx+max_node_idx+1 for edge_idx in hypergraph.edge}
  # Now the indices range from 1-(n+m)
  hypergraph = Relabel(hypergraph, node2inc, edge2inc)
  node2original = {node2inc[curr]: original for curr, original in node2original.items()}
  edge2original = {edge2inc[curr]: original for curr, original in edge2original.items()}

  with open(data_path, 'w') as data_file:
    for node_idx, node in hypergraph.node.items():
      for edge_idx in node.edges:
        data_file.write("{} {} {}\n".format(
          node_idx, edge_idx, 1 if is_weighted else ""
        ))
        if not only_one_side:
          data_file.write("{} {} {}\n".format(
            edge_idx, node_idx, 1 if is_weighted else ""
          ))
  with open(metadata_path, 'w') as meta_file:
    for node, original in node2original.items():
      meta_file.write("Replace {} with node_idx {}\n".format(node, original))
    for edge, original in edge2original.items():
      meta_file.write("Replace {} with edge_idx {}\n".format(edge, original))



def LoadMetadataMaps(metadata_path):
  node_map = {}
  edge_map = {}
  with open(metadata_path) as file:
    for line in file:
      tokens = line.split()
      assert len(tokens) == 5
      written_idx = int(tokens[1])
      original_idx = int(tokens[4])
      node_edge_switch = tokens[3]
      if node_edge_switch == 'node_idx':
        node_map[written_idx] = original_idx
      elif node_edge_switch == 'edge_idx':
        edge_map[written_idx] = original_idx
      else:
        raise ValueError("Metadata file is invalid")
  return node_map, edge_map

def LoadEdgeList(data_path, metadata_path):
  node_map, edge_map = LoadMetadataMaps(metadata_path)
  hypergraph = Hypergraph()
  with open(data_path) as file:
    for line in file:
      tokens = line.split()
      assert len(tokens) == 2 or len(tokens) == 3
      left_idx = int(tokens[0])
      right_idx = int(tokens[1])
      if left_idx in node_map:
        assert right_idx in edge_map
        AddNodeToEdge(hypergraph, node_map[left_idx], edge_map[right_idx])
      elif left_idx in edge_map:
        assert right_idx in node_map
        AddNodeToEdge(hypergraph, node_map[right_idx], edge_map[left_idx])
      else:
        raise ValueError("Hypergraph file is invalid. Idx {} not found.".format(left_idx))
  return hypergraph


PARSING_OPTIONS = {
    "AMINER":
        AMinerToHypergraph,
    "SNAP":
        SnapCommunityToHypergraph,
    "SNAP_CLEAN":
        lambda source: CleanHypergraph(SnapCommunityToHypergraph(source)),
    "DL_MAD_GRADES":
        DownloadMadGrades,
    "MTX":
        LoadMTX,
    "HMETIS":
        LoadHMetis,
}
