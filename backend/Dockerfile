FROM rust:latest

WORKDIR /usr/src/app

COPY Cargo.toml ./

RUN mkdir src && \
    echo "fn main(){}" > src/main.rs && \
    cargo build --release

COPY src ./src

RUN touch src/main.rs && cargo build --release

EXPOSE 55320

CMD ["./target/release/backend"]