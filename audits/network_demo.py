"""
=============================================================================
COMPUTER NETWORKS — Demonstration Module
=============================================================================

This module explicitly demonstrates key Computer Networks concepts as they
apply to the AI Text Detection system:

  1. TCP Socket Communication (Transport Layer — OSI Layer 4)
  2. HTTP Request/Response Protocol (Application Layer — OSI Layer 7)
  3. WebSocket Protocol (Full-Duplex over TCP — RFC 6455)
  4. DNS Resolution (resolving hostnames to IP addresses)
  5. HTTP Status Codes and Headers
  6. Client-Server Architecture

Each function is labeled with the relevant CN concept and OSI layer.
=============================================================================
"""

import socket
import http.client
import urllib.parse
import json
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# CN CONCEPT 1: TCP Socket — Transport Layer (OSI Layer 4)
# =============================================================================
def demo_tcp_socket_check(host: str = "127.0.0.1", port: int = 8000) -> dict:
    """
    Demonstrates raw TCP socket communication.

    TCP (Transmission Control Protocol) is the foundation of internet communication.
    It provides:
      - Reliable, ordered, error-checked delivery of data
      - Three-way handshake: SYN -> SYN-ACK -> ACK
      - Connection teardown: FIN -> FIN-ACK -> ACK

    This function opens a TCP socket to the API server to verify it is
    listening on the correct port — exactly what happens before every
    HTTP request your browser or client makes.

    OSI Layer: 4 (Transport Layer)
    Protocol: TCP/IP
    """
    result = {
        "cn_concept": "TCP Socket — OSI Layer 4 (Transport Layer)",
        "host": host,
        "port": port,
        "connection": None,
        "latency_ms": None
    }

    try:
        start = time.perf_counter()

        # Create a TCP/IP socket (AF_INET = IPv4, SOCK_STREAM = TCP)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3.0)

        # Perform TCP three-way handshake
        # (SYN sent -> SYN-ACK received -> ACK sent)
        connection_result = sock.connect_ex((host, port))
        latency_ms = (time.perf_counter() - start) * 1000

        if connection_result == 0:
            result["connection"] = "SUCCESS — TCP handshake completed"
            result["status"] = "Port is open and server is listening"
        else:
            result["connection"] = f"FAILED — error code {connection_result}"
            result["status"] = "Server not reachable on this port"

        result["latency_ms"] = round(latency_ms, 3)
        sock.close()

    except socket.timeout:
        result["connection"] = "TIMEOUT — no response within 3 seconds"
    except Exception as e:
        result["connection"] = f"ERROR — {str(e)}"

    logger.info(f"[CN: TCP Socket] {result['connection']} | Latency: {result.get('latency_ms')} ms")
    return result


# =============================================================================
# CN CONCEPT 2: HTTP Protocol — Application Layer (OSI Layer 7)
# =============================================================================
def demo_http_request(host: str = "127.0.0.1", port: int = 8000) -> dict:
    """
    Demonstrates the HTTP/1.1 Request-Response protocol.

    HTTP (HyperText Transfer Protocol) runs over TCP and defines:
      - Request methods: GET, POST, PUT, DELETE, PATCH, OPTIONS
      - Request/response headers: key-value metadata
      - Status codes: 1xx informational, 2xx success, 3xx redirect,
                      4xx client error, 5xx server error
      - Body: optional payload (JSON, form data, binary)

    This makes a raw HTTP GET request to the /docs endpoint and
    prints the full response headers and status line.

    OSI Layer: 7 (Application Layer)
    Protocol: HTTP/1.1 over TCP
    """
    result = {
        "cn_concept": "HTTP/1.1 Protocol — OSI Layer 7 (Application Layer)",
        "request": f"GET http://{host}:{port}/docs",
        "status_code": None,
        "response_headers": {},
        "protocol": "HTTP/1.1"
    }

    try:
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/docs", headers={
            "User-Agent": "AI-Text-Detector-NetworkDemo/1.0",
            "Accept": "text/html",
            "Connection": "keep-alive"   # CN: persistent TCP connection
        })
        response = conn.getresponse()

        result["status_code"] = response.status
        result["status_reason"] = response.reason
        result["response_headers"] = dict(response.getheaders())
        result["body_length_bytes"] = len(response.read())

        conn.close()

        logger.info(f"[CN: HTTP] Response: {response.status} {response.reason}")
        logger.info(f"[CN: HTTP] Headers received: {list(result['response_headers'].keys())}")

    except Exception as e:
        result["error"] = str(e)
        logger.warning(f"[CN: HTTP] Request failed: {e}")

    return result


# =============================================================================
# CN CONCEPT 3: HTTP POST with JSON body
# =============================================================================
def demo_http_post_predict(text: str, host: str = "127.0.0.1", port: int = 8000) -> dict:
    """
    Demonstrates HTTP POST request with JSON body — the standard way
    REST APIs receive structured data from clients.

    Key CN concepts:
      - Content-Type: application/json (MIME type header)
      - Transfer-Encoding: indicates how the body is delivered
      - Response: JSON payload parsed from the HTTP body
      - Async pattern: server returns task_id, client polls for result

    OSI Layer: 7 (Application Layer)
    Protocol: HTTP/1.1 POST
    """
    payload = json.dumps({"text": text})
    result = {
        "cn_concept": "HTTP POST + JSON — REST API over TCP/IP",
        "request_method": "POST",
        "endpoint": f"http://{host}:{port}/predict",
        "payload_bytes": len(payload),
        "content_type": "application/json"
    }

    try:
        conn = http.client.HTTPConnection(host, port, timeout=10)
        headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(payload)),
            "Accept": "application/json"
        }
        conn.request("POST", "/predict", body=payload, headers=headers)
        response = conn.getresponse()
        body = response.read().decode("utf-8")

        result["status_code"] = response.status
        result["response_json"] = json.loads(body) if body else {}
        conn.close()

        logger.info(f"[CN: HTTP POST] Status: {response.status} | Response: {result['response_json']}")

    except Exception as e:
        result["error"] = str(e)
        logger.warning(f"[CN: HTTP POST] Failed: {e}")

    return result


# =============================================================================
# CN CONCEPT 4: DNS Resolution — Network Layer (OSI Layer 3/7)
# =============================================================================
def demo_dns_resolution(hostname: str = "localhost") -> dict:
    """
    Demonstrates DNS (Domain Name System) resolution.

    DNS translates human-readable domain names to IP addresses.
    This is essential for distributed systems where services
    communicate by hostname (e.g., Docker Compose uses internal DNS
    to resolve 'redis', 'postgres', 'web' to container IPs).

    In our Docker Compose setup:
      - 'redis' resolves to the Redis container IP
      - 'postgres' resolves to the PostgreSQL container IP
      - 'web' resolves to the FastAPI container IP

    This is how containers find each other without hardcoding IPs.

    OSI Layer: 7 (Application Layer) / uses UDP Layer 4
    Port: 53 (DNS standard port)
    """
    result = {
        "cn_concept": "DNS Resolution — Domain Name -> IP Address",
        "hostname": hostname,
        "resolved_addresses": [],
        "docker_internal_dns_note": (
            "In Docker Compose, service names act as hostnames. "
            "Docker's embedded DNS server (127.0.0.11) resolves them to container IPs."
        )
    }

    try:
        addr_info = socket.getaddrinfo(hostname, None)
        ips = list(set(info[4][0] for info in addr_info))
        result["resolved_addresses"] = ips
        result["resolution_status"] = "SUCCESS"
        logger.info(f"[CN: DNS] '{hostname}' resolved to: {ips}")
    except socket.gaierror as e:
        result["resolution_status"] = f"FAILED — {e}"
        logger.warning(f"[CN: DNS] Resolution failed for '{hostname}': {e}")

    return result


# =============================================================================
# CN CONCEPT 5: Network Ports and Protocols Summary
# =============================================================================
def get_network_architecture_summary() -> dict:
    """
    Returns the full network architecture of the AI Text Detection system,
    mapping each service to its protocol, port, and OSI layer.

    This demonstrates understanding of how different services communicate
    in a distributed microservices architecture.
    """
    return {
        "cn_concept": "Network Architecture — Port and Protocol Mapping",
        "osi_model_reference": {
            "Layer 7 (Application)": "HTTP, WebSocket, SMTP, DNS",
            "Layer 6 (Presentation)": "TLS/SSL, JSON encoding",
            "Layer 5 (Session)": "WebSocket sessions, TCP sessions",
            "Layer 4 (Transport)": "TCP (reliable), UDP (fast, unreliable)",
            "Layer 3 (Network)": "IP addressing, routing",
            "Layer 2 (Data Link)": "Ethernet, MAC addressing",
            "Layer 1 (Physical)": "Cables, Wi-Fi signals"
        },
        "system_services": [
            {
                "service": "Nginx Reverse Proxy",
                "port": 80,
                "protocol": "HTTP/1.1 + WebSocket",
                "osi_layer": "Layer 7 (Application)",
                "role": "Load balancer, reverse proxy, WebSocket upgrade"
            },
            {
                "service": "FastAPI Web Server",
                "port": 8000,
                "protocol": "HTTP/1.1 + WebSocket (RFC 6455)",
                "osi_layer": "Layer 7 (Application)",
                "role": "REST API + WebSocket /ws/predict endpoint"
            },
            {
                "service": "Redis Message Broker",
                "port": 6379,
                "protocol": "RESP (Redis Serialization Protocol) over TCP",
                "osi_layer": "Layer 7 (Application) over Layer 4 (TCP)",
                "role": "Celery task queue — async message passing"
            },
            {
                "service": "PostgreSQL Database",
                "port": 5432,
                "protocol": "PostgreSQL wire protocol over TCP",
                "osi_layer": "Layer 7 (Application) over Layer 4 (TCP)",
                "role": "Active Learning feedback storage, MLflow backend"
            },
            {
                "service": "MLflow Tracking Server",
                "port": 5000,
                "protocol": "HTTP/1.1",
                "osi_layer": "Layer 7 (Application)",
                "role": "Experiment tracking, model registry"
            },
            {
                "service": "Prometheus Metrics",
                "port": 9090,
                "protocol": "HTTP (pull-based scraping)",
                "osi_layer": "Layer 7 (Application)",
                "role": "Time-series metrics collection from /metrics endpoint"
            },
            {
                "service": "Grafana Dashboard",
                "port": 3000,
                "protocol": "HTTP/1.1",
                "osi_layer": "Layer 7 (Application)",
                "role": "Visualization of Prometheus metrics"
            }
        ],
        "communication_patterns": {
            "Client -> Nginx (port 80)": "HTTP/HTTPS request-response (CN: L7 proxy)",
            "Nginx -> FastAPI (port 8000)": "HTTP reverse proxy with X-Forwarded-For header",
            "FastAPI -> Redis (port 6379)": "TCP socket, RESP protocol (task dispatch)",
            "Celery Worker -> Redis": "TCP poll for new tasks (RESP BLPOP command)",
            "FastAPI -> PostgreSQL": "TCP, PostgreSQL wire protocol (feedback storage)",
            "Prometheus -> FastAPI /metrics": "HTTP GET scrape every 15s",
            "WebSocket /ws/predict": "TCP upgrade, full-duplex streaming"
        }
    }


# =============================================================================
# Main: Run all demos
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("COMPUTER NETWORKS — AI Text Detection System Demo")
    print("=" * 70 + "\n")

    # Demo 1: TCP Socket
    print("--- DEMO 1: TCP Socket (OSI Layer 4) ---")
    result = demo_tcp_socket_check()
    print(json.dumps(result, indent=2))

    # Demo 2: DNS
    print("\n--- DEMO 2: DNS Resolution ---")
    result = demo_dns_resolution("localhost")
    print(json.dumps(result, indent=2))

    # Demo 3: HTTP GET
    print("\n--- DEMO 3: HTTP GET Request (OSI Layer 7) ---")
    result = demo_http_request()
    print(json.dumps(result, indent=2))

    # Demo 4: Network Architecture
    print("\n--- DEMO 4: Full Network Architecture Summary ---")
    result = get_network_architecture_summary()
    print(json.dumps(result, indent=2))
