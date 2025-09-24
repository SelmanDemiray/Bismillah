# Use the official Rust image as a base
FROM rust:latest

# Create a new empty shell project
WORKDIR /usr/src/app

# Copy the Cargo.toml file
COPY Cargo.toml ./

# Copy the source code
COPY src ./src

# Build the project
RUN cargo build --release

# Expose the port the app runs on
EXPOSE 8080

# Set the startup command to run the binary
CMD ["./target/release/backend"]

