#!/bin/bash
curl -X POST -H "Content-Type: application/json" -d '{"tweet":"I love this app!"}' http://localhost:5001/analyze
