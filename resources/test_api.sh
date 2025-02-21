#!/bin/bash
curl -X POST -H "Content-Type: application/json" -d '{"tweet":"I do not like you!"}' http://localhost:5001/analyze
