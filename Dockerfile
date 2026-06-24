# Use Node.js LTS version (Debian-based for easier native builds)
FROM node:20-bullseye-slim

# Install system dependencies for @tensorflow/tfjs-node
RUN apt-get update && apt-get install -y \\
    python3 \\
    make \\
    g++ \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies (ignoring scripts initially for speed, then building)
RUN npm install

# Copy project files
COPY . .

# Ensure required directories exist
RUN mkdir -p models reports results

# Set environment variables
ENV NODE_ENV=production
ENV PORT=3000

# Expose port
EXPOSE 3000

# Run the application
CMD ["npm", "start"]
