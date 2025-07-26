# --- Stage 1: Build the frontend app ---
FROM node:20 AS builder
WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

# --- Stage 2: Serve with NGINX ---
FROM nginx:stable-alpine
WORKDIR /usr/share/nginx/html

# Remove default static files
RUN rm -rf ./*

# Copy built frontend files
COPY --from=builder /app/dist .

# Copy dynamic NGINX config template
COPY nginx.conf /etc/nginx/templates/default.conf.template

# Cloud Run requires the container to listen on $PORT (default: 8080)
ENV PORT=8080

# Start NGINX
CMD ["nginx", "-g", "daemon off;"]
