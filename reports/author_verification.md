# Author verification against Google Scholar

Reference labels: S survey/tutorial/overview, M magazine overview (survey or magazine-overview accepted), R research (research or magazine-overview accepted), B book (must not be counted as a survey).

## Profile completeness

| Author   | Source                        |   Papers |   Citations |   h |   i10 | Scholar top-20 found   |
|----------|-------------------------------|----------|-------------|-----|-------|------------------------|
| hanzo    | Google Scholar                |          |     126,877 | 153 |  1440 | 20/20                  |
| hanzo    | Semantic Scholar              |    2,029 |      53,383 | 103 |   800 | 11/20                  |
| hanzo    | Semantic Scholar (merged IDs) |    2,710 |      88,856 | 131 |  1275 | 20/20                  |
| hanzo    | OpenAlex                      |    1,045 |      41,714 |  86 |   530 | 10/20                  |
| hanzo    | Combined                      |    2,604 |      85,694 | 129 |  1193 | 18/20                  |
| niyato   | Google Scholar                |          |     142,895 | 169 |  1460 | 20/20                  |
| niyato   | Semantic Scholar              |    1,264 |      54,524 | 111 |   599 | 14/20                  |
| niyato   | Semantic Scholar (merged IDs) |    2,514 |     101,777 | 143 |  1292 | 20/20                  |
| niyato   | OpenAlex                      |    2,262 |      96,652 | 139 |  1177 | 17/20                  |
| niyato   | Combined                      |    2,680 |     113,214 | 151 |  1372 | 20/20                  |
| yu       | Google Scholar                |          |      69,098 | 133 |   717 | 20/20                  |
| yu       | Semantic Scholar              |      501 |      19,864 |  78 |   280 | 9/20                   |
| yu       | Semantic Scholar (merged IDs) |      985 |      40,612 | 101 |   535 | 16/20                  |
| yu       | OpenAlex                      |    1,494 |      57,283 | 118 |   717 | 17/20                  |
| yu       | Combined                      |    1,515 |      58,227 | 118 |   726 | 17/20                  |
| han      | Google Scholar                |          |     125,835 | 166 |  1506 | 20/20                  |
| han      | Semantic Scholar              |    1,268 |      61,658 | 121 |   798 | 16/20                  |
| han      | Semantic Scholar (merged IDs) |    1,353 |      70,229 | 128 |   864 | 19/20                  |
| han      | OpenAlex                      |    2,534 |      94,705 | 139 |  1350 | 17/20                  |
| han      | Combined                      |    2,642 |     104,767 | 147 |  1426 | 18/20                  |
| hossain  | Google Scholar                |          |      55,626 | 119 |   485 | 20/20                  |
| hossain  | Semantic Scholar              |      663 |      37,096 | 100 |   392 | 19/20                  |
| hossain  | Semantic Scholar (merged IDs) |      733 |      40,090 | 107 |   422 | 19/20                  |
| hossain  | OpenAlex                      |      802 |      34,545 |  99 |   381 | 17/20                  |
| hossain  | Combined                      |      889 |      44,336 | 111 |   458 | 20/20                  |

## Classification of Scholar top-20 items

Items whose title is in the training set (`data/real_dataset.csv`) are counted only in the first column.

| Author   | Profile used                  | Correct (all found)   | Correct (not in training)   |
|----------|-------------------------------|-----------------------|-----------------------------|
| hanzo    | Semantic Scholar (merged IDs) | 20/20                 | 16/16                       |
| niyato   | Semantic Scholar (merged IDs) | 20/20                 | 10/10                       |
| yu       | Semantic Scholar (merged IDs) | 16/16                 | 14/14                       |
| han      | Semantic Scholar (merged IDs) | 19/19                 | 15/15                       |
| hossain  | Semantic Scholar (merged IDs) | 16/19                 | 11/14                       |
| Total    |                               | 91/94 (97%)           | 66/69 (96%)                 |

## Details

| Author   | Ref   | Classifier        |    | In training   | Title                                                                  |
|----------|-------|-------------------|----|---------------|------------------------------------------------------------------------|
| hanzo    | S     | survey            | ✓  | yes           | On the road to 6G: Visions, requirements, key technologies, and testbe |
| hanzo    | S     | survey            | ✓  |               | Towards 6G wireless communication networks: Vision, enabling technolog |
| hanzo    | S     | survey            | ✓  |               | Joint radar and communication design: Applications, state-of-the-art,  |
| hanzo    | B     | research          | ✓  |               | OFDM and MC-CDMA for broadband multi-user communications, WLANs and br |
| hanzo    | B     | research          | ✓  |               | Mobile radio communications                                            |
| hanzo    | S     | survey            | ✓  |               | A survey on wireless security: Technical challenges, recent advances,  |
| hanzo    | S     | survey            | ✓  | yes           | A survey of non-orthogonal multiple access for 5G                      |
| hanzo    | S     | survey            | ✓  |               | Spatial modulation for generalized MIMO: Challenges, opportunities, an |
| hanzo    | B     | research          | ✓  |               | Turbo coding, turbo equalisation and space-time coding                 |
| hanzo    | S     | survey            | ✓  |               | Nonorthogonal multiple access for 5G and beyond                        |
| hanzo    | M     | survey            | ✓  |               | Machine learning paradigms for next-generation wireless networks       |
| hanzo    | M     | magazine-overview | ✓  |               | Reconfigurable intelligent surfaces for 6G systems: Principles, applic |
| hanzo    | R     | research          | ✓  |               | Multicell MIMO communications relying on intelligent reflecting surfac |
| hanzo    | R     | research          | ✓  |               | MU-MIMO communications with MIMO radar: From co-existence to joint tra |
| hanzo    | S     | survey            | ✓  | yes           | Fifty years of MIMO detection: The road to large-scale MIMOs           |
| hanzo    | R     | research          | ✓  |               | Reconfigurable intelligent surface-based wireless communications: Ante |
| hanzo    | M     | magazine-overview | ✓  |               | Green radio: radio techniques to enable energy-efficient wireless netw |
| hanzo    | R     | research          | ✓  |               | Intelligent reflecting surface aided MIMO broadcasting for simultaneou |
| hanzo    | S     | survey            | ✓  | yes           | Millimeter-wave communications: Physical channel models, design consid |
| hanzo    | M     | survey            | ✓  |               | Orthogonal time-frequency space modulation: A promising next-generatio |
| niyato   | S     | survey            | ✓  | yes           | Federated learning in mobile edge networks: A comprehensive survey     |
| niyato   | S     | survey            | ✓  |               | A survey of mobile cloud computing: architecture, applications, and ap |
| niyato   | S     | survey            | ✓  | yes           | Wireless networks with RF energy harvesting: A contemporary survey     |
| niyato   | S     | survey            | ✓  | yes           | Applications of deep reinforcement learning in communications and netw |
| niyato   | S     | survey            | ✓  | yes           | Convergence of edge computing and deep learning: A comprehensive surve |
| niyato   | S     | survey            | ✓  |               | 6G Internet of Things: A comprehensive survey                          |
| niyato   | B     | research          | ✓  |               | Game theory in wireless and communication networks: theory, models, an |
| niyato   | S     | survey            | ✓  | yes           | A survey on software-defined networking                                |
| niyato   | S     | survey            | ✓  |               | A survey on consensus mechanisms and mining strategy management in blo |
| niyato   | S     | survey            | ✓  | yes           | Machine learning in wireless sensor networks: Algorithms, strategies,  |
| niyato   | S     | survey            | ✓  | yes           | Wireless charging technologies: Fundamentals, standards, and network a |
| niyato   | S     | survey            | ✓  | yes           | Toward smart wireless communications via intelligent reflecting surfac |
| niyato   | R     | research          | ✓  |               | Incentive mechanism for reliable federated learning: A joint optimizat |
| niyato   | S     | survey            | ✓  | yes           | Ambient backscatter communications: A contemporary survey              |
| niyato   | B     | research          | ✓  |               | Dynamic spectrum access and management in cognitive radio networks     |
| niyato   | S     | survey            | ✓  | yes           | Semantic communications for future internet: Fundamentals, application |
| niyato   | R     | research          | ✓  |               | Privacy-preserving traffic flow prediction: A federated learning appro |
| niyato   | S     | survey            | ✓  |               | Federated learning meets blockchain in edge computing: Opportunities a |
| niyato   | R     | research          | ✓  |               | Optimization of resource provisioning cost in cloud computing          |
| niyato   | S     | survey            | ✓  |               | Proof-of-stake consensus mechanisms for future blockchain networks: fu |
| yu       | S     | survey            | ✓  |               | Big data analytics in intelligent transportation systems               |
| yu       | S     | survey            | ✓  |               | Software-defined networking (SDN) and distributed denial of service (D |
| yu       | S     | not in profile    |    |               | Machine learning techniques applied to software defined networking (SD |
| yu       | S     | not in profile    |    |               | Blockchain technology applied to smart cities                          |
| yu       | S     | survey            | ✓  |               | Enabling massive IoT toward 6G                                         |
| yu       | M     | magazine-overview | ✓  |               | UAV-assisted emergency networks in disasters                           |
| yu       | S     | survey            | ✓  |               | Wireless network virtualization                                        |
| yu       | S     | survey            | ✓  |               | Integrated blockchain and edge computing systems                       |
| yu       | R     | research          | ✓  |               | Computation offloading and resource allocation in wireless cellular ne |
| yu       | S     | survey            | ✓  |               | In-band full-duplex relaying                                           |
| yu       | S     | not in profile    |    |               | Industrial Internet                                                    |
| yu       | S     | survey            | ✓  |               | A survey on cyber-security of connected and autonomous vehicles (CAVs) |
| yu       | S     | survey            | ✓  | yes           | Covert communications: A comprehensive survey                          |
| yu       | R     | research          | ✓  |               | A new method to support UMTS/WLAN vertical handover using SCTP         |
| yu       | R     | research          | ✓  |               | A distributed consensus-based cooperative spectrum-sensing scheme in c |
| yu       | S     | not in profile    |    |               | A survey on the scalability of blockchain systems                      |
| yu       | S     | survey            | ✓  |               | A survey on zero-knowledge proof in blockchain                         |
| yu       | R     | research          | ✓  |               | Energy-efficient resource allocation for heterogeneous cognitive radio |
| yu       | S     | survey            | ✓  |               | Applications of the Internet of Things (IoT) in smart logistics: A com |
| yu       | S     | survey            | ✓  | yes           | Blockchain and machine learning for communications and networking syst |
| han      | S     | survey            | ✓  | yes           | Wireless networks with RF energy harvesting: A contemporary survey     |
| han      | B     | not in profile    |    |               | Game theory in wireless and communication networks: theory, models, an |
| han      | R     | research          | ✓  |               | Improving wireless physical layer security via cooperating relays      |
| han      | M     | survey            | ✓  |               | Machine learning paradigms for next-generation wireless networks       |
| han      | S     | survey            | ✓  | yes           | Wireless charging technologies: Fundamentals, standards, and network a |
| han      | S     | survey            | ✓  |               | Coalitional game theory for communication networks                     |
| han      | S     | survey            | ✓  |               | Game-theoretic methods for the smart grid: An overview of microgrid sy |
| han      | S     | survey            | ✓  | yes           | Federated learning for internet of things: Recent advances, taxonomy,  |
| han      | S     | survey            | ✓  |               | Reconfigurable intelligent surfaces for wireless communications: Princ |
| han      | R     | research          | ✓  |               | A deep reinforcement learning network for traffic light cycle control  |
| han      | B     | research          | ✓  |               | Dynamic spectrum access and management in cognitive radio networks     |
| han      | R     | research          | ✓  |               | Detecting stealthy false data injection using machine learning in smar |
| han      | R     | research          | ✓  |               | Information theoretic framework of trust modeling and evaluation for a |
| han      | M     | magazine-overview | ✓  |               | Digital-twin-enabled 6G: Vision, architectural trends, and future dire |
| han      | R     | research          | ✓  |               | Fair multiuser channel allocation for OFDMA networks using Nash bargai |
| han      | R     | research          | ✓  |               | When mobile blockchain meets edge computing                            |
| han      | M     | magazine-overview | ✓  |               | Federated learning for edge networks: Resource optimization and incent |
| han      | S     | survey            | ✓  |               | Matching theory for future wireless networks: Fundamentals and applica |
| han      | S     | survey            | ✓  | yes           | Device fingerprinting in wireless networks: Challenges and opportuniti |
| han      | R     | research          | ✓  |               | Hybrid beamforming for reconfigurable intelligent surface based multi- |
| hossain  | B     | not in profile    |    |               | Introduction to network simulator 2 (NS2)                              |
| hossain  | R     | research          | ✓  |               | Enabling localized peer-to-peer electricity trading among plug-in hybr |
| hossain  | S     | survey            | ✓  | yes           | Machine learning in IoT security: Current solutions and future challen |
| hossain  | S     | survey            | ✓  | yes           | Stochastic geometry for modeling, analysis, and design of multi-tier a |
| hossain  | S     | survey            | ✓  | yes           | Federated learning for internet of things: Recent advances, taxonomy,  |
| hossain  | R     | research          | ✓  |               | Dynamic user clustering and power allocation for uplink and downlink n |
| hossain  | B     | research          | ✓  |               | Dynamic spectrum access and management in cognitive radio networks     |
| hossain  | M     | survey            | ✓  |               | Evolution toward 5G multi-tier cellular wireless networks: An interfer |
| hossain  | S     | magazine-overview | ✗  |               | 5G cellular: key enabling technologies and research challenges         |
| hossain  | R     | research          | ✓  |               | Competitive pricing for spectrum sharing in cognitive radio networks:  |
| hossain  | S     | survey            | ✓  |               | Evolution of NOMA toward next generation multiple access (NGMA) for 6G |
| hossain  | M     | survey            | ✓  |               | Random access for machine-to-machine communication in LTE-advanced net |
| hossain  | R     | research          | ✓  |               | Dynamics of network selection in heterogeneous wireless networks: An e |
| hossain  | R     | survey            | ✗  |               | Machine learning techniques for cooperative spectrum sensing in cognit |
| hossain  | S     | survey            | ✓  | yes           | Machine learning for resource management in cellular and IoT networks: |
| hossain  | R     | research          | ✓  |               | Resource allocation for spectrum underlay in cognitive radio networks  |
| hossain  | S     | survey            | ✓  | yes           | Single and multi-agent deep reinforcement learning for AI-enabled wire |
| hossain  | M     | survey            | ✓  |               | Interference management in OFDMA femtocell networks: Issues and approa |
| hossain  | R     | research          | ✓  |               | Resource allocation for device-to-device communications underlaying LT |
| hossain  | B     | survey            | ✗  |               | Cognitive wireless communication networks                              |
