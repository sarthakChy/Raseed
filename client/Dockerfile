# --- Stage 1: Build the frontend app ---
FROM node:20 AS builder
WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy the app and build
COPY . .
RUN npm run build

# --- Stage 2: Serve with NGINX on port 80 ---
FROM nginx:stable-alpine
WORKDIR /usr/share/nginx/html

# Clean default nginx files
RUN rm -rf ./*

# Copy build output from previous stage
COPY --from=builder /app/build .

# Add SPA-friendly routing
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
