# FlyLab literature fact sheet

Curated, short, cited. Each fact carries a source tag. If a claim is not here, treat it as
**unverified**. Wording is paraphrased from the papers' abstracts/main results. Facts tagged
[FlyLab-data] come from this repository's data and code, not from a paper.

Sources
- [Shiu2024] Shiu P.K., Sterne G.R., Spiller N., et al. "A Drosophila computational brain model reveals sensorimotor processing." Nature 634, 210–219 (2024). doi:10.1038/s41586-024-07763-9
- [Dorkenwald2024] Dorkenwald S., et al. "Neuronal wiring diagram of an adult brain." Nature 634, 124–138 (2024). doi:10.1038/s41586-024-07558-y
- [Schlegel2024] Schlegel P., et al. "Whole-brain annotation and multi-connectome cell typing of Drosophila." Nature 634, 139–152 (2024). doi:10.1038/s41586-024-07686-5
- [Eckstein2024] Eckstein N., et al. "Neurotransmitter classification from electron microscopy images at synaptic sites in Drosophila melanogaster." Cell 187, 2574–2594 (2024).
- [Bidaye2014] Bidaye S.S., Machacek C., Wu Y., Dickson B.J. "Neuronal control of Drosophila walking direction." Science 344, 97–101 (2014).
- [Bidaye2020] Bidaye S.S., et al. "Two brain pathways initiate distinct forward walking programs in Drosophila." Neuron 108, 469–485 (2020).
- [Rayshubskiy2020] Rayshubskiy A., et al. "Neural circuit mechanisms for steering control in walking Drosophila." bioRxiv (2020). doi:10.1101/2020.04.04.024703
- [Hampel2015] Hampel S., Franconville R., Simpson J.H., Seeds A.M. "A neural command circuit for grooming movement control." eLife 4:e08758 (2015).
- [WangChen2024] Wang-Chen S., et al. "NeuroMechFly v2: simulating embodied sensorimotor control in adult Drosophila." Nature Methods 21, 2353–2362 (2024).
- [EonRepo] github.com/eonsystemspbc/fly-brain (PyTorch and other backends of the Shiu et al. model).

## Connectome (FlyWire)
- The FlyWire whole-brain connectome of an adult female Drosophila contains ~139,000 neurons and ~50 million chemical synapses. [Dorkenwald2024]
- FlyWire neurons were annotated hierarchically (super class, class, sub-class, cell type); the annotation defines 8,453 cell types. [Schlegel2024]
- Neurotransmitter identity (acetylcholine, GABA, glutamate, etc.) of each FlyWire neuron was predicted from EM images. [Eckstein2024]
- The model in FlyLab uses FlyWire materialization v783: 138,639 neurons and 15,091,983 connected neuron pairs (Completeness_783.csv / Connectivity_783.parquet). [FlyLab-data]

## The whole-brain model (Shiu et al. 2024)
- The model is a leaky integrate-and-fire network containing every neuron of the FlyWire connectome; connectivity and synapse sign come only from the connectome and neurotransmitter predictions. [Shiu2024]
- Synaptic weights are proportional to synapse counts; acetylcholine synapses are treated as excitatory and GABA/glutamate as inhibitory. [Shiu2024]
- Parameters (from the published code): resting/reset potential -52 mV, threshold -45 mV, membrane time constant 20 ms, synaptic time constant 5 ms, refractory period 2.2 ms, synaptic delay 1.8 ms, 0.275 mV per synapse; stimulated neurons receive Poisson input. [Shiu2024 code: model.py]
- The model has no learning or plasticity: weights are fixed by the wiring diagram. [Shiu2024 code: model.py]
- In the model, unstimulated neurons are silent (no spontaneous activity), so any firing is evoked by the stimulus. [FlyLab-data]

## Sugar, feeding and MN9
- Computational activation of sugar-sensing or water-sensing gustatory neurons in the model predicts neurons that respond to tastes and are required for feeding initiation. [Shiu2024]
- Computational activation of neurons in the feeding region of the brain predicts which neurons elicit motor neuron firing; these predictions were tested with optogenetic activation and behaviour experiments. [Shiu2024]
- MN9 is a proboscis motor neuron used by Shiu et al. as the model readout of feeding initiation (proboscis extension). [Shiu2024]
- The MN9 neuron used in the published example notebook (root id 720575940660219265) is annotated in FlyWire as cell type CB0701; FlyLab's MN9 group is both CB0701 neurons. [FlyLab-data]

## Bitter and taste interactions
- Activating other classes of gustatory neurons in the model makes accurate predictions of how several taste modalities interact, giving circuit-level insight into aversive and appetitive taste processing. [Shiu2024]
- Bitter GRNs in FlyWire are annotated as cell_sub_class 'bitter' (cell types LB1a–d). [FlyLab-data]

## Walking: forward (P9), backward (MDN) and steering (DNa01/DNa02)
- Activation of the moonwalker descending neurons (MDN) makes flies walk backwards. [Bidaye2014]
- P9 descending neurons initiate forward walking (one of two brain pathways that start forward walking). [Bidaye2020]
- The neurons labelled 'P9' in the Eon fly-brain benchmark are annotated in FlyWire as cell type DNp09. [FlyLab-data, EonRepo]
- DNa02 activity predicts turning toward the same side (ipsilateral steering); DNa01 is also steering-related. [Rayshubskiy2020]

## Grooming and mechanosensation
- Computational activation of mechanosensory neurons predicts activation of a small set of neurons comprising the antennal grooming circuit, which does not overlap with the gustatory circuits. [Shiu2024]
- Johnston's organ mechanosensory neurons feed a command circuit for antennal grooming (interneurons aBN1/aBN2 and descending neurons aDN). [Hampel2015]
- FlyWire annotates 187 Johnston's organ JO-F neurons with cell_sub_class 'grooming'. FlyLab has not mapped the aBN/aDN names onto FlyWire cell types. [FlyLab-data]

## Body model
- NeuroMechFly v2 (flygym) is a MuJoCo-based biomechanical model of the adult fly for simulating embodied sensorimotor control, with leg controllers including CPG-based and hybrid turning controllers. [WangChen2024]
- In FlyLab, the mapping from brain output neurons to body commands is hand-designed (flylab/interface.py); it is not part of the connectome or of the Shiu et al. model. [FlyLab-data]

## Escape take-off (looming, LPLC2, LC4, giant fiber)
Sources: [vonReyn2014] von Reyn C.R., et al. "A spike-timing mechanism for action selection." Nature Neuroscience 17, 962–970 (2014). [Klapoetke2017] Klapoetke N.C., et al. "Ultra-selective looming detection from radial motion opponency." Nature 551, 237–241 (2017). [vonReyn2017] von Reyn C.R., et al. "Feature integration drives probabilistic behavior in the Drosophila escape response." Neuron 94, 1190–1204 (2017). [Ache2019] Ache J.M., et al. "Neural basis for looming size and velocity encoding in the Drosophila giant fiber escape pathway." Current Biology 29, 1073–1081 (2019).
- The giant fiber descending neuron drives the fast ("short-mode") escape take-off in response to looming stimuli. [vonReyn2014]
- LPLC2 visual projection neurons are highly selective for looming (approaching) stimuli. [Klapoetke2017]
- LC4 and LPLC2 visual projection neurons provide looming information to the giant fiber. [vonReyn2017, Ache2019]
- The giant fiber is annotated in FlyWire as cell type DNp01 (synonym "Giant Fiber"); FlyLab's group giant_fiber = both DNp01 neurons. [FlyLab-data]
- In the FlyLab model, LPLC2 stimulation at 50 Hz drives the giant fiber to ~95–118 Hz; sugar GRN stimulation leaves it at 0 Hz. [FlyLab-data]
- FlyLab's take-off is a scripted animation triggered when the giant fiber exceeds 50 Hz; NeuroMechFly v2 in flygym is a walking model and does not simulate flight. [FlyLab-data]

## Fly "driving a car"
- No primary source (paper, code or original video) was found for online claims that a fly brain drove a car; they appear only in blogs and news roundups. [FlyLab-data]
- FlyLab's car is an analogy: real descending-neuron outputs of the model (P9, MDN, DNa01/DNa02) are mapped by a hand-designed link to a simple car; manoeuvres are chosen by Claude or a planner, nothing learns. [FlyLab-data]

## Collision avoidance (LPLC1) and target tracking (LC10)
Sources: [TanakaClark2022] Tanaka R., Clark D.A. "Neural mechanisms to exploit positional geometry for collision avoidance." Current Biology 32(11), 2357–2374 (2022). [Ribeiro2018] Ribeiro I.M.A., Drews M., Bahl A., et al. "Visual projection neurons mediating directed courtship in Drosophila." Cell 174(3), 607–621 (2018).
- Walking flies slow down when small objects move back-to-front in the frontolateral visual field, a collision-avoidance behaviour. [TanakaClark2022]
- The visual projection neuron LPLC1 is necessary for this collision-avoidance behaviour (synaptic silencing), and optogenetic activation of LPLC1 elicits slowing. [TanakaClark2022]
- LC10 visual projection neurons respond preferentially to small moving objects; males with LC10 silenced cannot orient toward or stay near a female, and unilateral LC10 activation recapitulates orienting. [Ribeiro2018]
- In the FlyLab model (P9 100 Hz + LPLC1 of one side, 0.5 s, 4 trials), left LPLC1 at 25/50 Hz drives right DNa01 to 31/46 Hz with left DNa01 at 0 Hz (turn away), mirror-symmetric on the right; equal input on both sides cancels; the giant fiber stays near 0 Hz. Note: the literature reports LPLC1 -> slowing; turning away is a MODEL prediction. [FlyLab-data]
- In the FlyLab model, LC10a of one side drives DNa02 of the same side (left 25 Hz -> DNa02_L 66 Hz, DNa02_R 0 Hz): turning toward the stimulus. The effect is asymmetric (right LC10a 25 Hz -> DNa02_R only 35 Hz). [FlyLab-data]
- In the FlyLab model, unilateral stimulation of olfactory receptor neurons (DM1, DM4, VM2, DA2/geosmin, V/CO2) and of thermosensory neurons all produced the same ipsilateral DNa02 pattern regardless of valence, so the model does not support an "odour-valence steering" demo. [FlyLab-data]
- FlyLab's obstacle course (run_course) is closed loop: wall nearness per visual hemifield (designed eye model) drives LPLC1 of that side; the brain runs as one continuous trial in 20 ms chunks (BrainStepper, spike-identical to the one-shot kernel, scripts/validate_stepper.py). [FlyLab-data]

## Extra activities: swatter, bicycle, piano, Olympics, fly-throughs, sugar maze (FlyLab model data)
- In the FlyLab model (0.5 s, 4 trials, no walking), LPLC2 stimulation at 10/20/30/50 Hz drives the giant fiber to 33/64/85/105 Hz; LC4 alone at 10/20/30/50 Hz drives it to 0.5/8/26/47 Hz. Both groups get looming input from the literature (vonReyn2017, Ache2019); the relative strengths are MODEL measurements. [FlyLab-data]
- In the FlyLab model, bitter GRNs alone (200 Hz, 1 s, 8 trials) leave every curated output at 0 Hz, including MDN (backward walking): bitter does not make the model fly back up; it only suppresses sugar-evoked MN9. [FlyLab-data]
- FlyLab's looming eye model (flylab/looming.py) is designed: LPLC2 and LC4 Poisson rate = 100 Hz x clip(angular expansion speed / 500 deg/s, 0, 1). [FlyLab-data]
- Fly swatter (swat_fly, 4 cm pad starting 15 cm away, P9 walking, closed loop in 5 ms chunks): the intact model's giant fiber crosses 50 Hz while the swatter is still 112-134 mm away (17-20 deg wide) at every speed tested (250-3000 mm/s); the fly is off the ground 25-525 ms before impact. With LPLC2 silenced the giant fiber crosses threshold later (normal swat: 74 mm / 30 deg, margin 135 ms instead of 255 ms). With the giant fiber silenced there is no escape (splat). The ~17 deg escape angle is a MODEL prediction; it depends on the designed eye model. [FlyLab-data]
- Real flies also have a slower, giant-fiber-independent escape pathway (long-mode take-off) that FlyLab does not model. [vonReyn2014]
- In FlyLab closed-loop courses, wall input to both eyes' LPLC1 also drives the giant fiber weakly (forest run: mean 5.8 Hz, occasional 20-ms chunks above 50 Hz) even without any looming object. A bat "dodge" is therefore only counted while the bat's looming drive is >= 5 Hz; other bursts are reported separately. MODEL observation. [FlyLab-data]
- Sugar maze (run_maze; touching a pellet drives sugar GRNs, plus bitter GRNs if poisoned, 200 Hz; MN9 > 30 Hz = stop and eat): the intact model ate 3/3 sugar pellets and 0/2 poisoned pellets and finished the 136 m corridor; with bitter GRNs silenced it ate 3/3 sugar and 2/2 poisoned. Bitter suppressing sugar-evoked feeding matches Shiu2024; the maze itself is a FlyLab design. [FlyLab-data]
- Bicycle (ride_bike, "loop" preset): the physics-simulated legs walked 11-14 mm/s; the "left" manoeuvre (P9 + left DNa01/DNa02) produced +190 deg/s of leg yaw and turned the bike +186 deg; with left DNa01/DNa02 silenced the same manoeuvre produced -50 deg/s and the bike turned -47 deg (right). [FlyLab-data]
- Piano keyboard (play_piano keyboard): in 1.5 s of walking, front-foot touchdowns of three model flies (walk_forward, turn_left, walk_backward) played 32, 27 and 34 notes. The rhythm is flygym's CPG stepping, not a brain property. [FlyLab-data]
- Brain orchestra (play_piano orchestra, 8 bars at 100 BPM, one trial per musician fly): sugar GRN pulses gave MN9 a mean of 17 Hz per bar in sugar bars vs 3 Hz in the two bars where bitter GRNs were added (bitter mutes the bass); LPLC2 looming pulses on beats 2 and 4 drove giant-fiber bursts above 50 Hz on 28 eighth notes (snare), LC4 off-beat pulses gave 10-50 Hz on 20 (hi-hat). The score, chords and note mapping are FlyLab designs. [FlyLab-data]
- Singer fly (play_piano song_orchestra, Ode to Joy, one sugar pulse per note; MN9 read over each note's first 0.45 s = pulse + latency): 30/30 notes crossed the 20 Hz MN9 criterion (mean 68 Hz); with bitter GRNs added to the second phrase, those 15 notes fell to 8 Hz and went silent while the first phrase still sang (64 Hz). [FlyLab-data]
- Musical road (play_piano song_road, Ode to Joy, P9 100 Hz): a physics-simulated fly walking over the song's keys played 28/30 notes in order except one swapped pair, at ~73 BPM (8x slowed) set by its own stepping; the melody is the score laid on the floor, not a brain property. [FlyLab-data]
- Connectome Olympics (run_olympics, canyon): intact model finished in 20.3 s; with all LPLC1 silenced it crashed at 13% of the course, with left DNa01+DNa02 silenced at 8%, and with only the left eye's LPLC1 silenced at 5% (one-sided input biases steering continuously). MODEL predictions. [FlyLab-data]
- Fly-throughs (fly_through, closed loop, flight not simulated): cave 108 m in 20.7 s and forest 127 m in 24.4 s without touching a wall or obstacle; a head-on bat's looming drove the giant fiber to 75 Hz with the bat 6-8 m away (dodge, scripted). [FlyLab-data]
