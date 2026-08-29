# ---------- build ----------
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json ./
RUN npm install --no-audit --no-fund
COPY . .
# Header links only - all real traffic goes through the nginx proxy on this origin.
ARG VITE_N8N_URL=http://localhost:47678
ARG VITE_API_URL=http://localhost:47800
ARG VITE_MINIO_URL=http://localhost:47901
ENV VITE_N8N_URL=$VITE_N8N_URL \
    VITE_API_URL=$VITE_API_URL \
    VITE_MINIO_URL=$VITE_MINIO_URL
RUN npm run build

# ---------- serve ----------
FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
# Templates are rendered with envsubst on container start, so the same image
# works whether the AI service runs natively on the host or in the compose net.
COPY default.conf.template /etc/nginx/templates/default.conf.template
EXPOSE 80
# 127.0.0.1, NOT localhost: inside the container localhost resolves to ::1
# first, `listen 80;` only binds IPv4, and wget then reports "connection
# refused" forever - the container serves traffic perfectly but reports
# unhealthy, which makes `docker compose up --wait` hang.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD wget -q -O- http://127.0.0.1/ >/dev/null || exit 1
