import logging
import json
from typing import Optional
from .neptune_base import NeptuneBase

try:
    import boto3
    from langchain_aws import NeptuneAnalyticsGraph, NeptuneGraph
    from botocore.client import BaseClient
except ImportError:
    raise ImportError("The 'boto3' library is required. Please install it using 'pip install boto3'.")

logger = logging.getLogger(__name__)


class NeptuneGraphProxy:
    def __init__(
            self,
            graph_id: str,
            client: BaseClient,
            logger: Optional[logging.Logger] = None,
    ):
        """
        Abstraction layer to provide the same API compare toe LangChain AWS
        Constructs a NeptuneGraph object for AWS service interaction,
        with optional custom logger and boto client.
        """
        self.graph_id = graph_id
        self.client = client
        self.logger = logger or logging.getLogger(__name__)

    def query(
        self, query_string: str, parameter_map: Optional[dict] = None
    ):
        """
        Wrapper method to execute an incoming OpenCypher query
        and return the result from the Boto client.

        Args:
            query_string (str): OpenCypher query in string format.
            parameter_map (dict, optional): Parameter map for parameterized queries. Defaults to None.

        Returns:
            dict: Result from the Boto client.
        """
        query_params = {
            "graphIdentifier": self.graph_id,
            "queryString": query_string,
            "language": "OPEN_CYPHER",
            "parameters": {},
        }

        # Add parameters if provided
        if parameter_map:
            query_params["parameters"] = parameter_map

        self.logger.warning(
            f"Executing generic query with data [{query_params}] on graph [{self.graph_id}]"
        )
        response = self.client.execute_query(**query_params)  # type: ignore[attr-defined]

        return json.loads(response["payload"].read())["results"]


class MemoryGraph(NeptuneBase):
    def __init__(self, config):
        self.config = config

        self.graph = None
        endpoint = self.config.graph_store.config.endpoint
        if endpoint and endpoint.startswith("neptune-graph://"):
            graph_identifier = endpoint.replace("neptune-graph://", "")
            # Construct the object
            logger.warning("Constrcut object with proxy client.")
            client = boto3.client("neptune-graph")
            self.botoGraph = NeptuneGraphProxy(
                graph_id=graph_identifier,
                client=client
            )
            self.graph = NeptuneAnalyticsGraph(graph_identifier)

        if not self.graph:
            raise ValueError("Unable to create a Neptune client: missing 'endpoint' in config")

        self.node_label = ":`__Entity__`" if self.config.graph_store.config.base_label else ""

        self.embedding_model = NeptuneBase._create_embedding_model(self.config)

        self.llm_provider = "openai_structured"
        if self.config.llm.provider:
            self.llm_provider = self.config.llm.provider
        if self.config.graph_store.llm:
            self.llm_provider = self.config.graph_store.llm.provider

        self.llm = NeptuneBase._create_llm(self.config, self.llm_provider)
        self.user_id = None
        self.threshold = 0.7

    def _delete_entities_cypher(self, source, destination, relationship, user_id):
        """
        Returns the OpenCypher query and parameters for deleting entities in the graph DB

        :param source: source node
        :param destination: destination node
        :param relationship: relationship label
        :param user_id: user_id to use
        :return: str, dict
        """

        cypher = f"""
            MATCH (n {self.node_label} {{name: $source_name, user_id: $user_id}})
            -[r:{relationship}]->
            (m {self.node_label} {{name: $dest_name, user_id: $user_id}})
            DELETE r
            RETURN 
                n.name AS source,
                m.name AS target,
                type(r) AS relationship
            """
        params = {
            "source_name": source,
            "dest_name": destination,
            "user_id": user_id,
        }
        logger.debug(f"_delete_entities\n  query={cypher}")
        return cypher, params

    def _add_entities_cypher(
        self,
        source_node_list,
        source,
        source_embedding,
        source_type,
        destination_node_list,
        destination,
        dest_embedding,
        destination_type,
        relationship,
        user_id,
    ):
        """
        Returns the OpenCypher query and parameters for adding entities in the graph DB

        :param source_node_list: list of source nodes
        :param source: source node name
        :param source_embedding: source node embedding
        :param source_type: source node label
        :param destination_node_list: list of dest nodes
        :param destination: destination name
        :param dest_embedding: destination embedding
        :param destination_type: destination node label
        :param relationship: relationship label
        :param user_id: user id to use
        :return: str, dict
        """

        source_label = self.node_label if self.node_label else f":`{source_type}`"
        source_extra_set = f", source:`{source_type}`" if self.node_label else ""
        destination_label = self.node_label if self.node_label else f":`{destination_type}`"
        destination_extra_set = f", destination:`{destination_type}`" if self.node_label else ""

        # Refactor this code with the graph_memory.py implementation
        if not destination_node_list and source_node_list:
            cypher = f"""
                    MATCH (source)
                    WHERE id(source) = $source_id
                    SET source.mentions = coalesce(source.mentions, 0) + 1
                    WITH source
                    MERGE (destination {destination_label} {{name: $destination_name, user_id: $user_id}})
                    ON CREATE SET
                        destination.created = timestamp(),
                        destination.mentions = 1
                        {destination_extra_set}
                    ON MATCH SET
                        destination.mentions = coalesce(destination.mentions, 0) + 1
                    WITH source, destination, $dest_embedding as dest_embedding
                    CALL neptune.algo.vectors.upsert(destination, dest_embedding)
                    WITH source, destination
                    MERGE (source)-[r:{relationship}]->(destination)
                    ON CREATE SET 
                        r.created = timestamp(),
                        r.mentions = 1
                    ON MATCH SET
                        r.mentions = coalesce(r.mentions, 0) + 1
                    RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                    """

            params = {
                "source_id": source_node_list[0]["id(source_candidate)"],
                "destination_name": destination,
                "dest_embedding": dest_embedding,
                "user_id": user_id,
            }
        elif destination_node_list and not source_node_list:
            cypher = f"""
                    MATCH (destination)
                    WHERE id(destination) = $destination_id
                    SET destination.mentions = coalesce(destination.mentions, 0) + 1
                    WITH destination
                    MERGE (source {source_label} {{name: $source_name, user_id: $user_id}})
                    ON CREATE SET
                        source.created = timestamp(),
                        source.mentions = 1
                        {source_extra_set}
                    ON MATCH SET
                        source.mentions = coalesce(source.mentions, 0) + 1
                    WITH source, destination, $source_embedding as source_embedding
                    CALL neptune.algo.vectors.upsert(source, source_embedding)
                    WITH source, destination
                    MERGE (source)-[r:{relationship}]->(destination)
                    ON CREATE SET 
                        r.created = timestamp(),
                        r.mentions = 1
                    ON MATCH SET
                        r.mentions = coalesce(r.mentions, 0) + 1
                    RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                    """

            params = {
                "destination_id": destination_node_list[0]["id(destination_candidate)"],
                "source_name": source,
                "source_embedding": source_embedding,
                "user_id": user_id,
            }
        elif source_node_list and destination_node_list:
            cypher = f"""
                    MATCH (source)
                    WHERE id(source) = $source_id
                    SET source.mentions = coalesce(source.mentions, 0) + 1
                    WITH source
                    MATCH (destination)
                    WHERE id(destination) = $destination_id
                    SET destination.mentions = coalesce(destination.mentions) + 1
                    MERGE (source)-[r:{relationship}]->(destination)
                    ON CREATE SET 
                        r.created_at = timestamp(),
                        r.updated_at = timestamp(),
                        r.mentions = 1
                    ON MATCH SET r.mentions = coalesce(r.mentions, 0) + 1
                    RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                    """
            params = {
                "source_id": source_node_list[0]["id(source_candidate)"],
                "destination_id": destination_node_list[0]["id(destination_candidate)"],
                "user_id": user_id,
            }
        else:
            cypher = f"""
                    MERGE (n {source_label} {{name: $source_name, user_id: $user_id}})
                    ON CREATE SET n.created = timestamp(),
                                  n.mentions = 1
                                  {source_extra_set}
                    ON MATCH SET n.mentions = coalesce(n.mentions, 0) + 1
                    WITH n, $source_embedding as source_embedding
                    CALL neptune.algo.vectors.upsert(n, source_embedding)
                    WITH n
                    MERGE (m {destination_label} {{name: $dest_name, user_id: $user_id}})
                    ON CREATE SET m.created = timestamp(),
                                  m.mentions = 1
                                  {destination_extra_set}
                    ON MATCH SET m.mentions = coalesce(m.mentions, 0) + 1
                    WITH n, m, $dest_embedding as dest_embedding
                    CALL neptune.algo.vectors.upsert(m, dest_embedding)
                    WITH n, m
                    MERGE (n)-[rel:{relationship}]->(m)
                    ON CREATE SET rel.created = timestamp(), rel.mentions = 1
                    ON MATCH SET rel.mentions = coalesce(rel.mentions, 0) + 1
                    RETURN n.name AS source, type(rel) AS relationship, m.name AS target
                    """
            params = {
                "source_name": source,
                "dest_name": destination,
                "source_embedding": source_embedding,
                "dest_embedding": dest_embedding,
                "user_id": user_id,
            }
        logger.debug(
            f"_add_entities:\n  destination_node_search_result={destination_node_list}\n  source_node_search_result={source_node_list}\n  query={cypher}"
        )
        return cypher, params

    def _search_source_node_cypher(self, source_embedding, user_id, threshold):
        """
        Returns the OpenCypher query and parameters to search for source nodes

        :param source_embedding: source vector
        :param user_id: user_id to use
        :param threshold: the threshold for similarity
        :return: str, dict
        """
        cypher = f"""
            MATCH (source_candidate {self.node_label})
            WHERE source_candidate.user_id = $user_id 

            WITH source_candidate, $source_embedding as v_embedding
            CALL neptune.algo.vectors.distanceByEmbedding(
                v_embedding,
                source_candidate,
                {{metric:"CosineSimilarity"}}
            ) YIELD distance
            WITH source_candidate, distance AS cosine_similarity
            WHERE cosine_similarity >= $threshold

            WITH source_candidate, cosine_similarity
            ORDER BY cosine_similarity DESC
            LIMIT 1

            RETURN id(source_candidate), cosine_similarity
            """

        params = {
            "source_embedding": source_embedding,
            "user_id": user_id,
            "threshold": threshold,
        }
        logger.debug(f"_search_source_node\n  query={cypher}")
        return cypher, params

    def _search_destination_node_cypher(self, destination_embedding, user_id, threshold):
        """
        Returns the OpenCypher query and parameters to search for destination nodes

        :param source_embedding: source vector
        :param user_id: user_id to use
        :param threshold: the threshold for similarity
        :return: str, dict
        """
        cypher = f"""
                MATCH (destination_candidate {self.node_label})
                WHERE destination_candidate.user_id = $user_id
                
                WITH destination_candidate, $destination_embedding as v_embedding
                CALL neptune.algo.vectors.distanceByEmbedding(
                    v_embedding,
                    destination_candidate, 
                    {{metric:"CosineSimilarity"}}
                ) YIELD distance
                WITH destination_candidate, distance AS cosine_similarity
                WHERE cosine_similarity >= $threshold

                WITH destination_candidate, cosine_similarity
                ORDER BY cosine_similarity DESC
                LIMIT 1
    
                RETURN id(destination_candidate), cosine_similarity
                """
        params = {
            "destination_embedding": destination_embedding,
            "user_id": user_id,
            "threshold": threshold,
        }

        logger.debug(f"_search_destination_node\n  query={cypher}")
        return cypher, params

    def _delete_all_cypher(self, filters):
        """
        Returns the OpenCypher query and parameters to delete all edges/nodes in the memory store

        :param filters: search filters
        :return: str, dict
        """
        cypher = f"""
        MATCH (n {self.node_label} {{user_id: $user_id}})
        DETACH DELETE n
        """
        params = {"user_id": filters["user_id"]}

        logger.debug(f"delete_all query={cypher}")
        return cypher, params

    def _get_all_cypher(self, filters, limit):
        """
        Returns the OpenCypher query and parameters to get all edges/nodes in the memory store

        :param filters: search filters
        :param limit: return limit
        :return: str, dict
        """

        cypher = f"""
        MATCH (n {self.node_label} {{user_id: $user_id}})-[r]->(m {self.node_label} {{user_id: $user_id}})
        RETURN n.name AS source, type(r) AS relationship, m.name AS target
        LIMIT $limit
        """
        params = {"user_id": filters["user_id"], "limit": limit}
        return cypher, params

    def _search_graph_db_cypher(self, n_embedding, filters, limit):
        """
        Returns the OpenCypher query and parameters to search for similar nodes in the memory store

        :param n_embedding: node vector
        :param filters: search filters
        :param limit: return limit
        :return: str, dict
        """

        cypher_query = f"""
            MATCH (n {self.node_label})
            WHERE n.user_id = $user_id
            WITH n, $n_embedding as n_embedding
            CALL neptune.algo.vectors.distanceByEmbedding(
                n_embedding,
                n,
                {{metric:"CosineSimilarity"}}
            ) YIELD distance
            WITH n, distance as similarity
            WHERE similarity >= $threshold
            CALL {{
                WITH n
                MATCH (n)-[r]->(m) 
                RETURN n.name AS source, id(n) AS source_id, type(r) AS relationship, id(r) AS relation_id, m.name AS destination, id(m) AS destination_id
                UNION ALL
                WITH n
                MATCH (m)-[r]->(n) 
                RETURN m.name AS source, id(m) AS source_id, type(r) AS relationship, id(r) AS relation_id, n.name AS destination, id(n) AS destination_id
            }}
            WITH distinct source, source_id, relationship, relation_id, destination, destination_id, similarity
            RETURN source, source_id, relationship, relation_id, destination, destination_id, similarity
            ORDER BY similarity DESC
            LIMIT $limit
            """
        params = {
            "n_embedding": n_embedding,
            "threshold": self.threshold,
            "user_id": filters["user_id"],
            "limit": limit,
        }
        logger.debug(f"_search_graph_db\n  query={cypher_query}")

        return cypher_query, params
