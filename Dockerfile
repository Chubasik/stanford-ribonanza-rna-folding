FROM nvcr.io/nvidia/rapidsai/notebooks:23.10-cuda11.8-py3.10

# Consider using requirements.txt instead.
# I prefer to cache as many packages as I can.
RUN pip install -U torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
RUN pip install tensorflow==2.14.0
RUN pip install tensorboard==2.14.1

# hf
RUN pip install transformers==4.34.0
RUN pip install tokenizers==0.14.1
RUN pip install accelerate==0.23.0
RUN pip install datasets==2.14.5
RUN pip install peft==0.5.0
RUN pip install sentencepiece==0.1.99
RUN pip install sentence-transformers==2.2.2

RUN pip install plotly==5.17.0
RUN pip install seaborn==0.12.2
RUN pip install merlin-dataloader==23.8.0
RUN pip install wandb==0.15.12
RUN pip install kaggle==1.5.16
RUN pip install obonet==1.0.0
RUN pip install biopython==1.81
RUN pip install pyvis==0.3.2
RUN pip install apache-beam==2.51.0
RUN pip install mwparserfromhell==0.6.4
RUN pip install pywikibot==8.3.0
RUN pip install openai==0.27.8
RUN pip install Wikipedia-API==0.6.0
RUN pip install sympy==1.12
RUN pip install antlr4-python3-runtime==4.11.0
RUN pip install tiktoken==0.4.0
RUN pip install lxml==4.9.3
RUN pip install faiss-cpu==1.7.4
RUN pip install arnie==0.1.5
RUN pip install einops==0.7.0
# Eternafold
RUN mamba install -y -n base -c bioconda eternafold \
    && conda clean -afy
ENV ETERNAFOLD_PATH=/opt/conda/bin/eternafold-bin
ENV ETERNAFOLD_PARAMETERS=/opt/conda/lib/eternafold-lib/parameters/EternaFoldParams.v1
RUN pip install draw-rna==0.1.0
# CapR
USER root
RUN apt-get update && apt-get install -y git make g++
RUN git clone https://github.com/fukunagatsu/CapR.git /home/rapids/notebooks/toolkits/CapR && \
    cd /home/rapids/notebooks/toolkits/CapR && \
    make RIblast && \
    mv CapR /usr/local/bin/ && \
    chmod a+x /usr/local/bin/CapR
# RNAstructure
RUN wget -P /home/rapids/notebooks/toolkits/ http://rna.urmc.rochester.edu/Releases/current/RNAstructureLinuxTextInterfaces64bit.tgz && \
    cd /home/rapids/notebooks/toolkits/ && \
    tar -zxf RNAstructureLinuxTextInterfaces64bit.tgz && \
    echo "rnastructure: /home/rapids/notebooks/toolkits/RNAstructure/exe" >> /home/rapids/notebooks/toolkits/arnie.conf
RUN mkdir -p /home/rapids/notebooks/toolkits/tmp && \
    chmod 777 /home/rapids/notebooks/toolkits/tmp && \
    echo "TMP: /home/rapids/notebooks/toolkits/tmp" >> /home/rapids/notebooks/toolkits/arnie.conf
ENV DATAPATH=/home/rapids/notebooks/toolkits/RNAstructure/data_tables
# Vienna
RUN mamba install -y -n base -c bioconda viennarna && conda clean -afy && \
    echo "vienna_2: /opt/conda/bin/" >> /home/rapids/notebooks/toolkits/arnie.conf
# CONTRAfold
RUN mamba install -y -n base -c bioconda contrafold && conda clean -afy && \
    echo "contrafold_2: /opt/conda/bin/" >> /home/rapids/notebooks/toolkits/arnie.conf
# rnasoft
RUN apt-get update && apt-get install -y xutils-dev && \
    wget -P /home/rapids/notebooks/toolkits/ http://www.rnasoft.ca/download/MultiRNAFold-2.1.tar.gz && \
    cd /home/rapids/notebooks/toolkits/ && \
    tar -zxf MultiRNAFold-2.1.tar.gz && \
    cd MultiRNAFold && \
    make depend && \
    make && \
    chmod -R 777 /home/rapids/notebooks/toolkits/MultiRNAFold && \
    echo "rnasoft_07: /home/rapids/notebooks/toolkits/MultiRNAFold" >> /home/rapids/notebooks/toolkits/arnie.conf && \
    echo "rnasoft: /home/rapids/notebooks/toolkits/MultiRNAFold" >> /home/rapids/notebooks/toolkits/arnie.conf
# fix vienna bugs caused by parallel preprocessing
RUN sed -i "s/command = \['%s\/RNAfold' % LOC, '-p', '-T', str(T)\]/command = \['%s\/RNAfold' % LOC, '-p', '-T', str(T), '--noPS'\]/" /opt/conda/lib/python3.10/site-packages/arnie/pfunc.py
RUN sed -i "s/os.remove(\"%s_0001_ss.ps\" % output_id)/pass/" /opt/conda/lib/python3.10/site-packages/arnie/pfunc.py
RUN sed -i "s/command = \['%s\/RNAfold' % LOC, '-T', str(T), '-p0'\]/command = \['%s\/RNAfold' % LOC, '-T', str(T), '-p0', '--noPS'\]/" /opt/conda/lib/python3.10/site-packages/arnie/mfe.py

# Hotknots
RUN git clone https://github.com/deprekate/HotKnots.git /home/rapids/notebooks/toolkits/HotKnots && \
    cd /home/rapids/notebooks/toolkits/HotKnots && \
    git reset --hard 7694de40a2e5f2db0ee7947f7e41ca640033e213 && \
    pip install /home/rapids/notebooks/toolkits/HotKnots

# ipknot
RUN apt-get update && apt-get install -y musl-dev gcc g++ cmake make ninja-build git pkg-config zlib1g-dev
RUN git clone https://github.com/ERGO-Code/HiGHS /home/rapids/notebooks/toolkits/HiGHS \
    && cd /home/rapids/notebooks/toolkits/HiGHS \
    && cmake -DFAST_BUILD=ON -DCMAKE_BUILD_TYPE=Release -G Ninja -B build \
    && cmake --build build \
    && cmake --install build --strip
RUN ls /usr/local/lib/
RUN git clone https://github.com/satoken/ipknot.git /home/rapids/notebooks/toolkits/ipknot && \
    cd /home/rapids/notebooks/toolkits/ipknot && \
    cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_HIGHS=ON -DSTATIC_BUILD=OFF -DZLIB_LIBRARY=/usr/lib/x86_64-linux-gnu/libz.a -DZLIB_INCLUDE_DIR=/usr/include -G Ninja -B build && \
    cmake --build build && \
    cmake --install build --strip 
RUN echo "ipknot: /usr/local/bin/" >> /home/rapids/notebooks/toolkits/arnie.conf

# bpRNA
RUN mamba install -y -n base -c bioconda perl-graph
RUN git clone https://github.com/hendrixlab/bpRNA.git /home/rapids/notebooks/toolkits/bpRNA


ENV ARNIEFILE=/home/rapids/notebooks/toolkits/arnie.conf
USER root
ARG user_id=1000
RUN usermod -u ${user_id} rapids
USER rapids
