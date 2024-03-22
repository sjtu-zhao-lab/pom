# This Dockerfile configures a Docker environment that 
# contains all the required packages for the tool
FROM ubuntu:20.04
ARG UID
ARG GID
ARG VHLS_PATH
RUN echo "Group ID: $GID"
RUN echo "User ID: $UID"

USER root
RUN apt-get update -y && apt-get install apt-utils -y
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

# Install basic packages 
RUN apt-get upgrade -y 
RUN apt-get update -y \
    && apt-get install -y clang lld cmake libssl-dev\
                          pkg-config g++\
                          llvm gcc ninja-build \
                          build-essential autoconf libtool\
                          git vim wget sudo

CMD ["bash"]

# Add dev-user
# RUN groupadd -o -g $GID dev-user
# RUN useradd -r -g $GID -u $UID -m -d /home/dev-user -s /sbin/nologin -c "User" dev-user
# RUN echo "dev-user     ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
# USER dev-user

# Install PyTorch and Torch-MLIR
ENV PATH="${PATH}:~/.local/bin"
# RUN pip3 install --user --upgrade pip \
#     && pip3 install pandas dataclasses colorlog pyyaml


# Add environment variables
ENV vhls $VHLS_PATH
RUN printf "\
\nexport LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LIBRARY_PATH \
\n# Vitis HLS setup \
\nsource ${vhls}/Vitis/2022.2/settings64.sh \
\nsource ${vhls}/Vitis_HLS/2022.2/settings64.sh \
\nexport PATH=$PATH:/workspace/build/bin:/workspace/scalehls/polygeist/llvm/build/bin:/workspace/scalehls/polygeist/build/bin:~/.local/bin \
\n" >> ~/.vimrc
#Add vim environment
RUN printf "\
\nset autoread \
\nautocmd BufWritePost *.cpp silent! !clang-format -i <afile> \
\nautocmd BufWritePost *.c   silent! !clang-format -i <afile> \
\nautocmd BufWritePost *.h   silent! !clang-format -i <afile> \
\nautocmd BufWritePost *.hpp silent! !clang-format -i <afile> \
\nautocmd BufWritePost *.cc  silent! !clang-format -i <afile> \
\nautocmd BufWritePost *.py  silent! !python3 -m black <afile> \
\nautocmd BufWritePost *.sv  silent! !verible-verilog-format --inplace <afile> \
\nautocmd BufWritePost *.v  silent! !verible-verilog-format --inplace <afile> \
\nautocmd BufWritePost * redraw! \
\n" >> ~/.vimrc

# Entrypoint set up
WORKDIR /home/workspace 
# COPY . /usr/src/workspace
