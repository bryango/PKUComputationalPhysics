#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import http.server
import socketserver

PORT = 0  # chosen by system

Handler = http.server.SimpleHTTPRequestHandler
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(httpd.socket.getsockname()[1])
    httpd.serve_forever()
