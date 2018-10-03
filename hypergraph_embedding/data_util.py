# This file is responsible for preparing experimental data
from . import Hypergraph
from .hypergraph_util import *
from collections import namedtuple
import logging
import json
import requests
from tqdm import tqdm


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
    troubled_nodes = [node_idx
                      for node_idx, node in new_hg.node.items()
                      if len(node.edges) < min_degree]
    for node_idx in troubled_nodes:
      RemoveNode(new_hg, node_idx)
    troubled_edges = [edge_idx
                      for edge_idx, edge in new_hg.edge.items()
                      if len(edge.nodes) < min_degree]
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
    response = requests.get(instructor_url,
                            headers={"Authorization": "Token token={}".format(api_token)},
                            data={'page': page})
    instructor_json = json.loads(response.text)
    total_pages = instructor_json['totalPages']
    instructors_id_name = get_instructors_id_name(instructor_json)
    return instructors_id_name, total_pages

  def get_courses_for_instructor(instructor_id, page):
    response = requests.get(courses_url,
                            headers={"Authorization": "Token token={}".format(api_token)},
                            data={'instructor': instructor_id,
                                  'page': page})
    course_json = json.loads(response.text)
    courses_uuid_name = get_courses_uuid_name(course_json)
    total_pages = course_json['totalPages']
    return courses_uuid_name, total_pages

  instructors_id_name, total_pages = get_instructors_on_page(1)
  for page in tqdm(range(2, total_pages+1)):
    instructors_id_name.extend(get_instructors_on_page(page)[0])

  uuid_map = {}
  result = Hypergraph()
  for instructor_id, instructor_name in tqdm(instructors_id_name):
    result.node[instructor_id].name = instructor_name
    courses_uuid_name, total_pages = get_courses_for_instructor(instructor_id, 1)
    for page in range(2, total_pages+1):
      courses_uuid_name.extend(get_courses_for_instructor(instructor_id, page)[0])
    for course_uuid, course_name in courses_uuid_name:
      if course_uuid not in uuid_map:
        uuid_map[course_uuid] = len(uuid_map)
        if course_name is not None:
          result.edge[uuid_map[course_uuid]].name = course_name
      AddNodeToEdge(result, instructor_id, uuid_map[course_uuid])
  return result



PARSING_OPTIONS = {
    "AMINER": AMinerToHypergraph,
    "SNAP": SnapCommunityToHypergraph,
    "SNAP_CLEAN": lambda source: CleanHypergraph(SnapCommunityToHypergraph(source)),
    "DL_MAD_GRADES": DownloadMadGrades,
}
