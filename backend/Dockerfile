FROM rust:latest

WORKDIR /usr/src/app

COPY Cargo.toml ./

RUN mkdir src && \
    echo "fn main(){}" > src/main.rs && \
    cargo build --release

COPY src ./src

RUN touch src/main.rs && cargo build --release

# Create the data directory and set permissions
RUN mkdir -p /usr/src/app/data && chmod 777 /usr/src/app/data

EXPOSE 55320

CMD ["./target/release/backend"]