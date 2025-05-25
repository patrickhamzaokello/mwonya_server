FROM golang:alpine AS builder

# Install FFmpeg
RUN apk add --no-cache ffmpeg

WORKDIR /app
COPY go.mod ./
RUN go mod download

COPY . .
RUN go build -o streaming-server main.go

FROM alpine:latest
RUN apk add --no-cache ffmpeg ca-certificates

WORKDIR /app
COPY --from=builder /app/streaming-server .
COPY --from=builder /app/static ./static/
COPY --from=builder /app/process-audio.sh .
COPY --from=builder /app/scripts ./scripts/

# Create directories
RUN mkdir -p audio uploads

# Make scripts executable
RUN chmod +x process-audio.sh scripts/init-audio.sh

EXPOSE 8080

CMD ["./streaming-server"]