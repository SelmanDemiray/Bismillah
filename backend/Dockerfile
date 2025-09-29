# Dockerfile

# Use the official Rust image as a base
FROM rust:latest

# Create a new empty shell project
WORKDIR /usr/src/app

# Copy the Cargo files to cache dependencies
COPY Cargo.toml ./

# Create a dummy main.rs to build only the dependencies
RUN mkdir src && \
    echo "fn main(){}" > src/main.rs && \
    cargo build --release

# Copy the actual source code
COPY src ./src

# Build the project again with the cached dependencies
# This step will now be much faster
RUN cargo build --release

# Expose the port the app runs on
EXPOSE 8080

# Set the startup command to run the binary
CMD ["./target/release/backend"]