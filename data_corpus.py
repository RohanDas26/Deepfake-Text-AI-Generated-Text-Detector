"""
=============================================================================
DATA CORPUS — Comprehensive Multi-Domain Dataset for Text Deepfake Detection
=============================================================================
Classes:
  0: HUMAN-WRITTEN
  1: FULLY AI-GENERATED
  2: HUMAN-POLISHED / AI-PARAPHRASED

Domains Covered:
  - Technical / Programming / Engineering
  - Academic & Scientific Research
  - Conversational & Friendly Chat (ChatGPT / Claude / Gemini)
  - Personal Anecdotes & Daily Reflections
  - Journalism & News Analysis
  - Creative Fiction & Literature
  - Business & Professional Communication
=============================================================================
"""

CORPUS = [   {   'domain': 'technical_postmortem',
        'label': 0,
        'text': "Honestly, I wasn't expecting the server to launch on the first try. Spent three grueling hours "
                'debugging an edge-case null pointer inside the connection pool handler. Turns out the database driver '
                'silently drops packets when connection keep-alives timeout unexpectedly.'},
    {   'domain': 'personal_narrative',
        'label': 0,
        'text': 'My grandmother always used to bake sourdough on crisp Sunday mornings. The entire house would smell '
                "of burnt crust and melted salted butter. I still haven't managed to replicate her starter culture, no "
                'matter how many jars I spoil.'},
    {   'domain': 'academic_philosophy',
        'label': 0,
        'text': 'Epistemological skepticism does not merely reject empirical certitude; it fundamentally questions '
                'whether doxastic justifications can withstand radical Agrippan trilemmas without collapsing into '
                'infinite regress or dogmatic circularity.'},
    {   'domain': 'software_engineering',
        'label': 0,
        'text': 'We rolled back the Kubernetes cluster deployment within four minutes of receiving PagerDuty alerts. '
                'CPU saturation breached ninety-five percent across every pod in the replica pool. A rogue regex in '
                'the authentication middleware had triggered exponential catastrophic backtracking.'},
    {   'domain': 'casual_forum',
        'label': 0,
        'text': 'Wait, did anyone else notice the sound mixing in that scene was completely garbage? You could barely '
                'hear the dialogue over the ridiculous bass rumble. Either my soundbar is dying or the audio engineer '
                'gave up halfway through.'},
    {   'domain': 'journalism_commentary',
        'label': 0,
        'text': 'Walking down Market Street at dusk, the tension is palpable beneath the fluorescent glare of '
                'convenience stores. Local shopkeepers have taken to locking their doors even during peak afternoon '
                'hours. City hall promises reforms, but the street level reality tells a starkly different story.'},
    {   'domain': 'creative_fiction',
        'label': 0,
        'text': 'Rain lashed against the rusted corrugated roof with relentless fury. Silas gripped his coffee mug, '
                'cold porcelain biting into his calloused fingers. The lighthouse beam swept through the fog, catching '
                'nothing but empty whitecaps.'},
    {   'domain': 'business_email',
        'label': 0,
        'text': "Hey Sarah, quick heads up on tomorrow's client presentation. The deck looks mostly solid, but slide "
                "seven still references Q2 revenue targets instead of Q3 actuals. Let's patch that before the nine AM "
                "sync so we don't catch unnecessary grief."},
    {   'domain': 'scientific_biology',
        'label': 0,
        'text': 'During our field trials in the subalpine meadow, bumblebee visitation rates dropped by nearly forty '
                'percent following sudden unseasonal temperature spikes. Floral nectar volumes were visibly depleted, '
                'suggesting thermal stress directly impairs microvascular floral secretory tissue.'},
    {   'domain': 'humor_opinion',
        'label': 0,
        'text': 'Nothing tests your will to live quite like assembling flat-pack Swedish furniture at eleven PM on a '
                'Tuesday. I am missing exactly one proprietary wooden dowel, and my sanity has officially left the '
                'building.'},
    {   'domain': 'history_biography',
        'label': 0,
        'text': "Churchill's radio broadcasts during the Blitz were crafted with obsessive care. He paced his "
                'underground cabinet war rooms for hours, dictating monosyllabic cadences intended to resonate with '
                'working-class dockworkers and aristocrats alike.'},
    {   'domain': 'travel_reflection',
        'label': 0,
        'text': 'The overnight train out of Varanasi lurched violently every few kilometers, tossing passengers '
                'against the vinyl berths. Chai vendors pushed their metal kettles down narrow corridors, chanting '
                'their melodic wares into the humid pre-dawn mist.'},
    {   'domain': 'personal_frustration',
        'label': 0,
        'text': 'Why do airline websites make it virtually impossible to redeem flight credits? Every button redirects '
                "in an endless loop back to the homepage. I've spent forty-five minutes on hold listening to distorted "
                'elevator jazz.'},
    {   'domain': 'sports_journalism',
        'label': 0,
        'text': 'With three seconds remaining and trailing by two, Jenkins hoisted an off-balance three-pointer from '
                'thirty feet out. The ball kissed the glass, rattled around the iron rim twice, and somehow trickled '
                'through as the buzzer sounded. Pure pandemonium erupted.'},
    {   'domain': 'culinary_review',
        'label': 0,
        'text': 'The pasta was unmistakably hand-rolled—delicate yet possessing that requisite al dente bite. However, '
                "the ragu was oversalted to the point of overpowering the wild boar's natural earthiness. A "
                'disappointing misstep from a kitchen of this pedigree.'},
    {   'domain': 'academic_linguistics',
        'label': 0,
        'text': 'Syntactic movement constraints in polysynthetic languages pose notable counterexamples to traditional '
                'configurational generative models. Ergative agreement morphology frequently displays split-ergativity '
                'governed entirely by aspectual tiers rather than structural case.'},
    {   'domain': 'personal_childhood',
        'label': 0,
        'text': 'We used to build rickety tree forts out of stolen construction pallets behind the gravel quarry. If '
                'you stepped wrong on the ladder, rusty nails would bite right through your sneakers. Looking back, '
                "it's a small miracle none of us broke an arm."},
    {   'domain': 'hardware_engineering',
        'label': 0,
        'text': 'Soldering surface-mount capacitors without a decent microscope is pure masochism. The ground plane '
                'sucked away all the heat from my iron, creating nasty cold solder joints that took two days of '
                'continuity testing to trace.'},
    {   'domain': 'political_opinion',
        'label': 0,
        'text': 'Voter turnout in municipal school board elections remains embarrassingly low, rarely topping twelve '
                'percent. Yet these are the very boards making crucial budgetary decisions on school lunches, teacher '
                'pay, and curriculum standards.'},
    {   'domain': 'creative_poetry_prose',
        'label': 0,
        'text': 'Autumn arrived without fanfare. The maple outside my bedroom window shed its crimson canopy in a '
                'single gusty afternoon, leaving bare branches clawing at a slate-gray October sky.'},
    {   'domain': 'casual_recommendation',
        'label': 0,
        'text': "If you're ever in Seattle, skip the tourist trap clam chowder spot at the market and head three "
                'blocks uphill to the tiny bakery next to the comic shop. Their cardamom buns will genuinely change '
                'your perspective on life.'},
    {   'domain': 'software_architecture',
        'label': 0,
        'text': "Don't fall into the trap of premature microservice adoption. Monoliths are vastly easier to reason "
                'about, profile, and refactor until your engineering headcount genuinely requires decoupled release '
                'cadences.'},
    {   'domain': 'personal_health',
        'label': 0,
        'text': 'Started doing physical therapy for my lower back last month. The exercises look embarrassingly '
                'simple—mostly bridging and pelvic tilts—but doing them with strict form leaves me sweating profusely '
                'after ten minutes.'},
    {   'domain': 'astronomy_amateur',
        'label': 0,
        'text': "Set up my eight-inch Dobsonian telescope in the driveway despite the streetlights. Saturn's rings "
                'were crystal clear through the nine millimeter eyepiece. Even after twenty years of stargazing, '
                "seeing Cassini's division in real time takes my breath away."},
    {   'domain': 'parenting_reflection',
        'label': 0,
        'text': "My four-year-old asked me why the moon doesn't fall down if gravity pulls everything to the ground. "
                'Explaining orbital mechanics and centrifugal balance using an apple and a piece of yarn at seven in '
                'the morning was not on my bingo card.'},
    {   'domain': 'book_critique',
        'label': 0,
        'text': "The protagonist's sudden character redemption in the final act feels completely unearned. Three "
                "hundred pages of ruthless self-interest don't evaporate because of a twenty-second conversation with "
                'a dying stranger.'},
    {   'domain': 'office_politics',
        'label': 0,
        'text': 'Management just announced another mandatory return-to-office mandate, citing collaborative synergy. '
                'Funny how collaborative synergy magically disappears whenever executives schedule their remote summer '
                'Fridays.'},
    {   'domain': 'gardening_notes',
        'label': 0,
        'text': 'Aphids completely decimated my heirloom tomato seedlings over the weekend. I refuse to spray '
                "synthetic pesticides, so I've released five hundred ladybugs into the greenhouse. Nature can fight "
                'this one out.'},
    {   'domain': 'gaming_discussion',
        'label': 0,
        'text': 'The boss fight in phase three is downright cheap. The dodge window is literally two frames, and the '
                'camera locks behind a pillar every time he initiates his sweep attack. Pure artificial difficulty.'},
    {   'domain': 'nostalgia_memoir',
        'label': 0,
        'text': 'Remember rewinding cassette tapes with a yellow hexagonal pencil to save the Walkman batteries? We '
                'accepted absurd inconveniences back then without a second thought.'},
    {   'domain': 'broken_non_native_english',
        'label': 0,
        'text': 'yesterday me and my brother was going to market and buyed two apple for eat but it was raining very '
                'hard so we ran back home quickly'},
    {   'domain': 'run_on_stream_of_thought',
        'label': 0,
        'text': 'i was just walking down the road and then i saw this dog and it started barking at me so i got scared '
                'and crossed the street'},
    {   'domain': 'informal_internet_slang',
        'label': 0,
        'text': 'yo bro wtf was that movie ending so trash ngl like who even wrote that garbage screenplay fr'},
    {   'domain': 'all_caps_rant',
        'label': 0,
        'text': 'WHY ARE YOU ALWAYS DOING THIS TO ME I SWEAR TO GOD IT MAKES ABSOLUTELY ZERO SENSE'},
    {   'domain': 'campus_life',
        'label': 0,
        'text': 'The Wi-Fi in the library drops every time someone uses the microwave in the staff room. I thought my '
                'network card was failing until I literally correlated the disconnect timestamps with lunch breaks.'},
    {   'domain': 'qualitative_healthcare_research',
        'label': 0,
        'text': 'We ran thirty-six interviews with emergency department nurses across four municipal hospitals. The '
                "overriding sentiment wasn't frustration with patient volume, but rather the sheer cognitive friction "
                'of poorly designed electronic health record interfaces.'},
    {   'domain': 'devops_cautionary',
        'label': 0,
        'text': "Look, if you're going to use Redis as a primary datastore, you'd better have your append-only file "
                'persistence configured down to the millisecond. We learned that the hard way during an unscheduled '
                'datacenter power outage.'},
    {   'domain': 'family_anecdote',
        'label': 0,
        'text': 'My dad still keeps a tattered road atlas in the glove compartment of his Subaru. He refuses to trust '
                "Google Maps completely, insisting that GPS satellites can't anticipate washed-out gravel detours."},
    {   'domain': 'music_critique',
        'label': 0,
        'text': "The rhythm of jazz drumming isn't about rigid metronomic precision; it lives in the subtle push and "
                'pull against the bassline. That slight hesitation before the downbeat is where all the emotional '
                'swing actually happens.'},
    {   'domain': 'weekend_narrative',
        'label': 0,
        'text': 'I spent the better part of Saturday afternoon trying to coax a terrified stray kitten out from under '
                'the patio deck with tuna flakes. She finally ventured out, hissed at me indignantly, and sprinted '
                'straight into the hydrangeas.'},
    {   'domain': 'epistemological_philosophy',
        'label': 0,
        'text': "Kant's distinction between analytic and synthetic a priori judgments remains arguably the most "
                'pivotal hinge in modern epistemological philosophy. Without it, the entire edifice of rationalist '
                'metaphysics crumbles under Humean empiricism.'},
    {   'domain': 'career_advice',
        'label': 0,
        'text': "Don't let recruiters convince you that leetcode grinding is the only way to land a decent engineering "
                'gig. Building two production apps with actual users taught me ten times more about system design than '
                'memorizing inverted binary trees.'},
    {   'domain': 'coffee_review',
        'label': 0,
        'text': 'The espresso was sour enough to strip enamel off your teeth. Whatever bean origin they were boasting '
                'about on the chalkboard, the barista clearly under-extracted the shot at barely eighty degrees.'},
    {   'domain': 'travel_ordeal',
        'label': 0,
        'text': "My flight got diverted to Omaha because of lightning over O'Hare. Spent six hours sleeping on a vinyl "
                'bench between two vending machines before they finally found an available flight crew.'},
    {   'domain': 'local_civics',
        'label': 0,
        'text': 'The municipal water bond passed by less than seventy votes after three contentious town hall debates. '
                'It just goes to show how every single local ballot actually matters when neighborhood infrastructure '
                'is on the line.'},
    {   'domain': 'parenting_humor',
        'label': 0,
        'text': 'Trying to get three toddlers dressed in winter snowsuits is an Olympic sport in its own right. As '
                'soon as you zip up the second boot on one, another announces they urgently need the bathroom.'},
    {   'domain': 'pc_hardware',
        'label': 0,
        'text': 'We replaced the thermal paste on the graphics card and dropped load temperatures by nearly fourteen '
                "degrees Celsius. The manufacturer's factory application looked like dried cake frosting smeared "
                'haphazardly across the die.'},
    {   'domain': 'sensory_travel',
        'label': 0,
        'text': 'Walking through the open-air spice market in Marrakech hits you in waves—first dry cumin, then sharp '
                'crushed peppermint, followed by the deep warmth of toasted cinnamon bark.'},
    {   'domain': 'agile_workplace',
        'label': 0,
        'text': 'Our weekly sprint retrospective descended into a twenty-minute debate over whether Jira story points '
                'should represent ideal engineering hours or abstract cognitive complexity. Some rituals never '
                'change.'},
    {   'domain': 'atmospheric_fiction',
        'label': 0,
        'text': 'The old iron bridge groaned under the weight of the coal train as it crept across the river gorge. '
                'Beneath the rusted trestles, dark spring runoff churned white around jagged limestone boulders.'},
    {   'domain': 'human_discord_rant',
        'label': 0,
        'text': 'bruh my gpu drivers crashed three times mid game tonight and completely bricked my discord stream. '
                "had to roll back to the august release just so windows wouldn't blue screen every time i tabbed out. "
                'nvidia please fix your garbage updates.'},
    {   'domain': 'human_reddit_college',
        'label': 0,
        'text': 'Honestly so burnt out from this semester already. We have two midterms on Thursday, a compiler lab '
                "due Friday midnight, and my team partner hasn't replied to a single text since last Tuesday. Counting "
                'down the days until winter break.'},
    {   'domain': 'human_casual_cooking',
        'label': 0,
        'text': 'Tried making homemade ramen broth from scratch today. Simmered pork bones with roasted ginger and '
                'scallions for literally seven hours. Kitchen smells amazing but my feet are killing me and now I have '
                'a mountain of greasy pots to wash.'},
    {   'domain': 'human_personal_growth',
        'label': 0,
        'text': 'Turned 22 yesterday. Looking back at who I was two years ago, I barely recognize that kid. Still '
                "don't have my dream job lined up, but at least I finally stopped caring what random people on "
                'Instagram think about my life.'},
    {   'domain': 'human_coffee_chat',
        'label': 0,
        'text': 'Found this hole-in-the-wall coffee place tucked behind the metro station this morning. The barista '
                'drew a little cat in the latte foam and the cinnamon bun was fresh out of the oven. Made my whole '
                'gloomy Monday morning so much better.'},
    {   'domain': 'human_gym_struggle',
        'label': 0,
        'text': 'Hit a new personal record on deadlifts this morning—315 lbs for two clean reps! Then immediately '
                'dropped my shaker bottle and spilled chocolate protein powder all over the gym rubber mat. Instant '
                'karma keeping me humble lmao.'},
    {   'domain': 'human_work_frustration',
        'label': 0,
        'text': 'Love it when management schedules a ninety-minute meeting to discuss why our sprints are running '
                'behind schedule. If we spent half as much time writing code as we do talking about velocity charts, '
                'the feature would have shipped three weeks ago.'},
    {   'domain': 'human_book_thoughts',
        'label': 0,
        'text': "Just finished reading East of Eden and I'm honestly still recovering. Steinbeck's characterization of "
                'Cathy is genuinely terrifying, but the whole Timshel monologue at the end gave me actual chills. '
                "Easily the best book I've read all year."},
    {   'domain': 'human_parenting_casual',
        'label': 0,
        'text': 'My four-year-old just asked me where the sun goes when it sleeps and then immediately burst into '
                "tears because I said it doesn't sleep in a bed with a blanket. Toddler logic is an absolute "
                'rollercoaster and I need more caffeine.'},
    {   'domain': 'human_gaming_nostalgia',
        'label': 0,
        'text': 'Booted up my old Nintendo DS lite yesterday to see if the battery still held a charge. Found my '
                'Pokémon Platinum save file from 2009 with my Empoleon nicknamed Bubbles. The wave of pure middle '
                'school nostalgia hit like a truck.'},
    {   'domain': 'human_apartment_hunting',
        'label': 0,
        'text': 'Apartment hunting in this city is absolute nightmare fuel. Visited a studio today where the shower '
                'was literally inside the kitchen next to the mini fridge, and the landlord had the audacity to ask '
                'for three months deposit upfront with a straight face.'},
    {   'domain': 'human_creative_musings',
        'label': 0,
        'text': 'There is this weird thirty-minute window right before dusk when the streetlights flick on but the sky '
                'is still indigo. The entire neighborhood goes completely quiet for a second before the evening '
                'traffic picks up. Always my favorite part of walking home.'},
    {   'domain': 'human_hardware_mishap',
        'label': 0,
        'text': "PSA: do not try to clean your mechanical keyboard while it's still plugged in. Just accidentally sent "
                'a four-line string of random gibberish to my boss on Slack and locked my workstation keyboard layout '
                'to Dvorak.'},
    {   'domain': 'human_music_discovery',
        'label': 0,
        'text': "Algorithm randomly recommended a 70s Japanese city pop playlist while I was doing dishes and now I've "
                'been down this rabbit hole for three hours straight. The basslines on these tracks are ridiculously '
                'groovy.'},
    {   'domain': 'human_weekend_recap',
        'label': 0,
        'text': 'Spent the entire Sunday afternoon helping my dad build raised garden beds in the backyard. Sunburned '
                'shoulders, dirt under my fingernails, and sore hamstrings, but seeing the first batch of tomato '
                'seedlings in the soil felt surprisingly rewarding.'},
    {   'domain': 'ai_tech_essay',
        'label': 1,
        'text': 'Artificial intelligence continues to transform modern industry through automated data processing and '
                'predictive analytics. By leveraging deep neural networks, organizations can uncover actionable '
                'insights from vast datasets. Furthermore, these technological advancements foster operational '
                'efficiency across diverse economic sectors.'},
    {   'domain': 'ai_academic_intro',
        'label': 1,
        'text': 'In recent years, renewable energy systems have gained substantial attention as viable alternatives to '
                'fossil fuels. Solar photovoltaic technology, in particular, offers significant potential for reducing '
                'carbon emissions. Consequently, international policy frameworks increasingly emphasize clean energy '
                'transition goals.'},
    {   'domain': 'ai_business_proposal',
        'label': 1,
        'text': 'Our proposed software solution delivers comprehensive integration capabilities tailored to modern '
                'enterprise workflows. By standardizing communication protocols across departments, the platform '
                'enhances cross-functional collaboration. Ultimately, this leads to improved project turnaround times '
                'and measurable cost savings.'},
    {   'domain': 'ai_health_advice',
        'label': 1,
        'text': 'Maintaining a balanced lifestyle requires a holistic combination of proper nutrition, regular '
                'exercise, and adequate sleep. Incorporating nutrient-rich whole foods supports cardiovascular health '
                'and boosts immune function. Moreover, consistent physical activity mitigates stress and improves '
                'overall mental well-being.'},
    {   'domain': 'ai_customer_service',
        'label': 1,
        'text': 'Thank you for contacting our customer support team regarding your recent order inquiry. We understand '
                'the importance of timely delivery and apologize for any inconvenience caused. Please rest assured '
                'that our team is actively investigating this matter to provide an expedient resolution.'},
    {   'domain': 'ai_environmental_summary',
        'label': 1,
        'text': 'Climate change represents one of the most critical challenges facing our global ecosystem today. '
                'Rising global temperatures contribute to severe weather events and altered precipitation patterns '
                'worldwide. Therefore, collaborative multilateral efforts are paramount to achieving long-term '
                'sustainability.'},
    {   'domain': 'ai_education_overview',
        'label': 1,
        'text': 'Digital learning platforms have fundamentally altered the landscape of contemporary higher education. '
                'Students can now access educational resources asynchronously from virtually any geographical '
                'location. In addition, interactive multimedia modules foster deeper conceptual comprehension.'},
    {   'domain': 'ai_philosophy_summary',
        'label': 1,
        'text': 'The concept of free will has long captivated the attention of moral philosophers and cognitive '
                'scientists alike. Deterministic viewpoints argue that human choices are inevitably dictated by prior '
                'causal conditions. Conversely, compatibilist frameworks attempt to reconcile moral accountability '
                'with scientific naturalism.'},
    {   'domain': 'ai_financial_analysis',
        'label': 1,
        'text': 'Diversification serves as a fundamental pillar of prudent risk management in contemporary investment '
                'portfolios. By allocating assets across uncorrelated asset classes, investors can reduce overall '
                'portfolio volatility. As a result, long-term capital preservation is effectively enhanced.'},
    {   'domain': 'ai_travel_guide',
        'label': 1,
        'text': 'Kyoto is renowned for its rich cultural heritage, featuring centuries-old temples and meticulously '
                'manicured gardens. Visitors can immerse themselves in traditional Japanese ceremonies while exploring '
                'scenic bamboo groves. Consequently, the city offers an unforgettable blend of historical charm and '
                'serene natural beauty.'},
    {   'domain': 'ai_cybersecurity_overview',
        'label': 1,
        'text': 'Implementing multi-factor authentication is essential for securing modern organizational digital '
                'infrastructure. This security protocol requires users to provide multiple verification credentials '
                'before gaining system access. Hence, unauthorized breaches resulting from compromised passwords are '
                'substantially minimized.'},
    {   'domain': 'ai_creative_story',
        'label': 1,
        'text': 'The quiet town of Oakridge was nestled gently between rolling emerald hills and a tranquil '
                'crystalline river. Every morning, the local clock tower sounded six resonant chimes to welcome the '
                'rising dawn. The townspeople took pride in their peaceful community, fostering traditions passed down '
                'through generations.'},
    {   'domain': 'ai_productivity_guide',
        'label': 1,
        'text': 'Effective time management is a critical skill for achieving sustainable professional success. '
                'Utilizing structured prioritization frameworks like the Eisenhower Matrix allows individuals to focus '
                'on impactful tasks. Consequently, productivity increases while workplace fatigue is substantially '
                'diminished.'},
    {   'domain': 'ai_scientific_methodology',
        'label': 1,
        'text': 'The double-blind experimental design remains the gold standard in contemporary clinical research '
                'methodology. By preventing both participants and researchers from knowing group assignments, '
                'experimental bias is effectively controlled. Consequently, the resulting empirical data possesses '
                'robust internal validity.'},
    {   'domain': 'ai_marketing_copy',
        'label': 1,
        'text': 'Experience unparalleled audio clarity with our next-generation wireless active noise-canceling '
                'headphones. Engineered with precision acoustic drivers, they deliver immersive sound quality and '
                'all-day comfort. Elevate your everyday listening journey with advanced technology designed for modern '
                'lifestyles.'},
    {   'domain': 'ai_art_history',
        'label': 1,
        'text': 'The Renaissance period marked a profound rebirth of classical art, humanistic philosophy, and '
                'scientific inquiry across Europe. Master artists pioneered techniques such as linear perspective and '
                'chiaroscuro to achieve unprecedented realism. These foundational innovations forever altered Western '
                'artistic expression.'},
    {   'domain': 'ai_smart_cities',
        'label': 1,
        'text': 'Smart city initiatives leverage Internet of Things sensor networks to optimize urban infrastructure '
                'management. Traffic light timings dynamically adjust to alleviate congestion based on real-time '
                'vehicular flow patterns. In doing so, urban centers reduce emissions while improving resident '
                'mobility.'},
    {   'domain': 'ai_nutrition_facts',
        'label': 1,
        'text': 'Adequate hydration plays an indispensable role in maintaining optimal physiological and metabolic '
                'function. Water facilitates cellular nutrient transport, regulates internal core temperature, and '
                'supports kidney filtration. Therefore, health practitioners universally recommend consistent fluid '
                'intake throughout the day.'},
    {   'domain': 'ai_cloud_computing',
        'label': 1,
        'text': 'Cloud computing architectures provide scalable computing resources on a flexible pay-as-you-go '
                'operational model. Organizations can rapidly provision virtual machines without investing in upfront '
                'physical hardware procurement. Consequently, businesses achieve greater agility in deploying novel '
                'digital services.'},
    {   'domain': 'ai_literature_review',
        'label': 1,
        'text': 'The existing scholarly literature indicates a strong correlation between early childhood reading '
                'habits and subsequent cognitive development. Exposure to diverse vocabularies enhances linguistic '
                'fluency and fosters empathetic perspective-taking. Thus, structured literacy programs demonstrate '
                'lasting pedagogical benefits.'},
    {   'domain': 'ai_robotics_future',
        'label': 1,
        'text': 'Advancements in autonomous robotics are revolutionizing modern warehouse logistics and supply chain '
                'operations. Automated guided vehicles navigate storage facilities using sophisticated computer vision '
                'and spatial mapping algorithms. This integration significantly accelerates order fulfillment '
                'workflows.'},
    {   'domain': 'ai_leadership_post',
        'label': 1,
        'text': 'True leadership extends beyond managing operational processes; it entails inspiring team members '
                'toward a shared visionary goal. Cultivating psychological safety empowers employees to propose '
                'innovative solutions without fear of retribution. In turn, high-performing organizational cultures '
                'thrive.'},
    {   'domain': 'ai_renewable_tech',
        'label': 1,
        'text': 'Wind turbine efficiency has increased dramatically over the past two decades due to aerodynamic blade '
                'design improvements. Modern offshore installations capitalize on stronger and more consistent oceanic '
                'wind patterns. As a result, unit energy generation costs continue to decline steadily.'},
    {   'domain': 'ai_mental_health',
        'label': 1,
        'text': 'Mindfulness-based cognitive therapy has emerged as an efficacious intervention for managing chronic '
                'stress and anxiety. By cultivating non-judgmental awareness of present-moment experiences, '
                'individuals learn to break maladaptive cognitive thought patterns. Empirical studies substantiate '
                'these psychological outcomes.'},
    {   'domain': 'ai_blockchain_intro',
        'label': 1,
        'text': 'Blockchain technology introduces decentralized cryptographic ledgers that ensure transparent and '
                'immutable record-keeping. Transactions are validated collectively by distributed network nodes rather '
                'than relying on a centralized intermediary. This architectural paradigm holds vast transformative '
                'potential.'},
    {   'domain': 'ai_sleep_science',
        'label': 1,
        'text': 'During rapid eye movement sleep, the brain actively consolidates episodic memories and processes '
                'emotional experiences. Disrupted circadian sleep cycles can impair cognitive executive function and '
                'weaken physiological recovery. Prioritizing structured sleep hygiene is therefore essential for '
                'long-term health.'},
    {   'domain': 'ai_quantum_computing',
        'label': 1,
        'text': 'Quantum computing exploits quantum mechanical phenomena such as superposition and entanglement to '
                'execute complex computations. Unlike classical binary bits, qubits can represent probabilistic '
                'combinations of zero and one simultaneously. This theoretical capability could revolutionize '
                'cryptographic modeling.'},
    {   'domain': 'ai_agriculture_tech',
        'label': 1,
        'text': 'Precision agriculture employs satellite imaging and soil moisture sensors to optimize fertilizer and '
                'irrigation deployment. By treating agricultural fields as heterogeneous environments, farmers '
                'minimize environmental runoff while maximizing crop yields. This methodology contributes '
                'significantly to global food security.'},
    {   'domain': 'ai_conclusion_paragraph',
        'label': 1,
        'text': 'In conclusion, the integration of digital tools into modern society presents both unprecedented '
                'opportunities and multifaceted challenges. While technological innovation enhances operational '
                'productivity, thoughtful regulatory oversight remains essential. Navigating this dynamic landscape '
                'requires ongoing interdisciplinary dialogue.'},
    {   'domain': 'ai_intro_template',
        'label': 1,
        'text': "In today's fast-paced digital era, the ability to adapt to rapid technological change has become more "
                'vital than ever before. Individuals and organizations alike must embrace continuous learning to '
                'remain competitive. This article will delve into the key strategies necessary for thriving in a '
                'dynamic global economy.'},
    {   'domain': 'ai_text_no_punctuation',
        'label': 1,
        'text': 'artificial intelligence systems optimize operational efficiency by automating complex data processing '
                'workflows across decentralized enterprise cloud infrastructure'},
    {   'domain': 'ai_concise_definition',
        'label': 1,
        'text': 'machine learning algorithms extract statistical representations from training corpora to perform '
                'automated classification tasks with high empirical accuracy'},
    {   'domain': 'ai_bullet_flow_unpunctuated',
        'label': 1,
        'text': 'cloud computing architectures deliver scalable virtualization resources on demand while mitigating '
                'initial capital expenditure for expanding technology organizations'},
    {   'domain': 'ai_academic_buzzwords',
        'label': 1,
        'text': 'deep neural architectures utilize non-linear activation functions to approximate complex continuous '
                'mappings between multi-dimensional feature representations'},
    {   'domain': 'ai_corporate_agility',
        'label': 1,
        'text': "In today's fast-paced corporate environment, effective cross-functional collaboration has become "
                'paramount to organizational agility. By leveraging integrated project management platforms, '
                'distributed teams can seamlessly coordinate deliverables. Furthermore, establishing transparent '
                'communication protocols significantly reduces redundant workflows.'},
    {   'domain': 'ai_ml_explainer',
        'label': 1,
        'text': 'Certainly! Machine learning algorithms can be categorized into supervised, unsupervised, and '
                'reinforcement learning paradigms. Supervised techniques utilize labeled datasets to optimize '
                'predictive mappings between input features and target variables. In contrast, unsupervised approaches '
                'discover latent geometric structures without explicit supervision.'},
    {   'domain': 'ai_biodiversity_essay',
        'label': 1,
        'text': 'Biodiversity conservation plays an indispensable role in maintaining planetary ecological '
                'equilibrium. Diverse ecosystems demonstrate greater resilience against climatic shocks and pest '
                'outbreaks. Consequently, international multilateral treaties increasingly prioritize the preservation '
                'of endangered habitats worldwide.'},
    {   'domain': 'ai_seo_guide',
        'label': 1,
        'text': "To optimize your website for search engines, ensure that your content directly addresses the user's "
                'search intent. Incorporating descriptive header tags and descriptive meta descriptions enhances '
                'indexing efficiency. Additionally, improving page load speeds contributes to higher organic '
                'visibility and lower bounce rates.'},
    {   'domain': 'ai_ev_transition',
        'label': 1,
        'text': 'The transition toward electric vehicles marks a transformative milestone in modern decarbonization '
                'strategies. Advances in lithium-ion battery chemistry continue to expand driving range while reducing '
                'production costs. Therefore, automotive manufacturers are rapidly reallocating capital expenditures '
                'toward electrification platforms.'},
    {   'domain': 'ai_psychotherapy_summary',
        'label': 1,
        'text': 'Cognitive behavioral therapy is a structured psychotherapeutic intervention that focuses on '
                'identifying maladaptive thought patterns. By replacing negative cognitive distortions with objective '
                'behavioral responses, patients achieve improved emotional regulation. Numerous clinical trials '
                'confirm the efficacy of this modality.'},
    {   'domain': 'ai_microservices_overview',
        'label': 1,
        'text': 'Microservice architectures decompose complex software systems into independently deployable, modular '
                'services. Each microservice encapsulates specific business capabilities and communicates via '
                'lightweight network APIs. This decoupled paradigm empowers development teams to scale components '
                'dynamically based on real-time traffic.'},
    {   'domain': 'ai_nuclear_conclusion',
        'label': 1,
        'text': 'In conclusion, the renaissance of nuclear energy presents a viable solution for baseline zero-carbon '
                'electricity generation. While initial capital costs and regulatory approvals remain substantial '
                'hurdles, modern modular reactor designs offer enhanced safety parameters. Achieving ambitious climate '
                'targets will necessitate an all-inclusive energy portfolio.'},
    {   'domain': 'ai_personal_finance',
        'label': 1,
        'text': 'Effective financial planning requires individuals to establish a clear budget and maintain a '
                'diversified emergency fund. Allocating consistent monthly savings toward low-cost index funds '
                'harnesses the exponential power of compound interest. Over long investment horizons, disciplined '
                'capital accumulation delivers reliable wealth preservation.'},
    {   'domain': 'ai_vertical_farming',
        'label': 1,
        'text': 'Urban vertical farming represents an innovative agricultural technique that optimizes food production '
                'within metropolitan areas. By cultivating crops in vertically stacked layers under controlled '
                'environmental conditions, growers conserve water and minimize transportation emissions. This model '
                'significantly enhances local food security.'},
    {   'domain': 'ai_ethics_discourse',
        'label': 1,
        'text': 'The role of ethics in artificial intelligence development cannot be overstated. Developing automated '
                'decision-making systems without robust fairness constraints risks amplifying existing societal '
                'biases. Thus, academic and industrial stakeholders must collaborate to implement transparent '
                'governance frameworks.'},
    {   'domain': 'ai_nlp_progress',
        'label': 1,
        'text': 'Natural language processing models have achieved human-level performance across a broad spectrum of '
                'benchmark tasks. By processing extensive linguistic corpora through self-attention mechanisms, '
                'transformers capture complex contextual nuances. As a result, automated translation and document '
                'summarization have reached unprecedented fidelity.'},
    {   'domain': 'ai_supply_chain',
        'label': 1,
        'text': 'Developing resilient supply chains requires modern enterprises to diversify supplier networks across '
                'multiple geographical regions. Implementing real-time tracking systems improves visibility across '
                'logistics channels from procurement to delivery. In doing so, organizations mitigate the catastrophic '
                'impact of regional supply disruptions.'},
    {   'domain': 'ai_telehealth_overview',
        'label': 1,
        'text': 'Telemedicine has revolutionized modern healthcare delivery by facilitating remote clinical '
                'consultations via secure digital platforms. Patients in rural communities gain direct access to '
                'specialized medical practitioners without the burden of long-distance travel. Consequently, overall '
                'patient outcomes and diagnostic turnaround times improve markedly.'},
    {   'domain': 'ai_mindfulness_guide',
        'label': 1,
        'text': 'It is widely recognized that regular mindfulness meditation cultivates sustained attentional control '
                'and reduces physiological stress markers. By practicing mindful breathing techniques daily, '
                'individuals activate parasympathetic nervous pathways. This biological response fosters cognitive '
                'resilience amidst demanding professional schedules.'},
    {   'domain': 'ai_ar_future',
        'label': 1,
        'text': 'Augmented reality technology bridges physical environments and virtual data by overlaying contextual '
                'digital assets onto real-time camera feeds. From industrial maintenance assistance to interactive '
                'educational modules, spatial computing introduces immersive user interfaces. As hardware '
                'miniaturization advances, widespread commercial adoption is projected to accelerate.'},
    {   'domain': 'ai_conversational_life_advice',
        'label': 1,
        'text': 'You know, sometimes I feel like one of the most interesting phases of life is the one where you know '
                "you want something big, but you haven't quite figured out how to get there yet. You're studying, "
                'building skills, trying new things, making plans, and occasionally wondering whether all this effort '
                'is actually taking you somewhere. The funny thing is, life rarely gives you a clear indication that '
                "you're making progress. But here's something worth thinking about: you don't necessarily need to have "
                'your entire future figured out at eighteen or twenty. You just need to keep becoming someone your '
                "future self would be proud of. Don't let the pressure make you forget to enjoy the journey."},
    {   'domain': 'ai_conversational_explainer_api',
        'label': 1,
        'text': 'Think of an API like a waiter at a busy restaurant. You are the customer sitting at the table looking '
                "at the menu, and the kitchen is the backend system preparing your order. You wouldn't just march into "
                'the kitchen and start grabbing ingredients yourself. Instead, you tell the waiter what you want, the '
                'waiter communicates your order to the kitchen, and then brings the prepared dish right back to your '
                'table. That is essentially how client-server communication works in web development.'},
    {   'domain': 'ai_conversational_motivation_study',
        'label': 1,
        'text': 'If you are feeling completely overwhelmed by your exam preparation right now, take a deep breath and '
                'pause for a moment. It is completely normal to experience anxiety when you have multiple deadlines '
                'looming ahead. You do not need to master the entire syllabus in a single afternoon. Break your study '
                'material into manageable twenty-minute blocks, focus on one key concept at a time, and remember that '
                'consistent small efforts add up to tremendous progress over time.'},
    {   'domain': 'ai_conversational_coding_frustration',
        'label': 1,
        'text': 'Debugging can definitely feel like banging your head against a brick wall, especially when you have '
                'spent three hours tracking down a bug that turns out to be a misplaced semicolon. When you find '
                'yourself getting frustrated, the best move is actually to step away from your computer. Grab a glass '
                'of water, stretch your legs, and give your mind a quick reset. Nine times out of ten, the solution '
                'clicks the moment you return with fresh eyes.'},
    {   'domain': 'ai_conversational_morning_routine',
        'label': 1,
        'text': 'Creating a solid morning routine does not mean waking up at five in the morning and running a '
                'marathon. It simply means starting your day with intention rather than immediately reaching for your '
                'smartphone and scrolling through stressful notifications. Even ten minutes of quiet coffee, light '
                'stretching, or writing down your top three priorities can completely transform your focus and '
                'emotional well-being throughout the day.'},
    {   'domain': 'ai_conversational_chat_intro',
        'label': 1,
        'text': 'Of course! I would be delighted to help you explore that topic. Whether you want to dive deep into '
                'technical system design, discuss career strategies, or simply bounce creative ideas back and forth, '
                'we can tailor the conversation entirely to your goals. Feel free to ask whatever is on your mind, and '
                'we will take it step by step.'},
    {   'domain': 'ai_conversational_learning_curve',
        'label': 1,
        'text': 'Learning any new technical framework always feels intimidating during the first couple of weeks. You '
                'will encounter confusing syntax, cryptic compiler errors, and moments where nothing seems to work as '
                'expected. Remember that every senior engineer you look up to was once in the exact same position. '
                'Mastery is not about never making mistakes; it is about building the resilience to keep '
                'troubleshooting until things make sense.'},
    {   'domain': 'ai_conversational_productivity_burnout',
        'label': 1,
        'text': 'Productivity is not about filling every single waking hour with relentless tasks until you burn out. '
                'True sustainable productivity is about knowing when to push hard and when to deliberately unplug and '
                'recharge your mental batteries. If you find your focus slipping and simple tasks taking twice as long '
                'as usual, that is your brain signaling that it desperately needs rest, not more coffee.'},
    {   'domain': 'ai_conversational_creative_storytelling',
        'label': 1,
        'text': 'The old cafe at the corner of Maple Street had seen three generations of dreamers, writers, and '
                'hurried commuters pass through its weathered oak doors. Steam hissed continuously from the espresso '
                'machine as afternoon sunlight spilled across worn vinyl booths. For Lucas, this was not just a coffee '
                'shop; it was the one sanctuary where he could quietly sketch the architectural outlines of the city '
                'he hoped to rebuild.'},
    {   'domain': 'ai_conversational_tech_reflection',
        'label': 1,
        'text': 'It is fascinating to observe how rapidly artificial intelligence has transitioned from theoretical '
                'computer science research into everyday consumer tools. Tools that once required specialized GPU '
                'clusters and Ph.D. expertise are now accessible through a simple web interface. While this '
                'democratization unlocks incredible creative possibilities, it also challenges us to critically '
                're-evaluate our definitions of authorship, intellectual property, and authentic human expression.'},
    {   'domain': 'ai_conversational_casual_decision',
        'label': 1,
        'text': 'When you are torn between two good career opportunities, trying to analyze every possible permutation '
                'can easily trap you in analysis paralysis. A helpful framework is to ask yourself which choice offers '
                'the steepest learning curve and connects you with higher-caliber mentors. Compensation and titles '
                'will naturally follow when you consistently place yourself in environments that force you to expand '
                'your capabilities.'},
    {   'domain': 'ai_conversational_fitness_advice',
        'label': 1,
        'text': 'Starting a fitness journey does not require complicated workout splits or extreme dietary '
                'restrictions. The most effective routine is simply the one that you can consistently sustain for six '
                'months without feeling miserable. Start by walking thirty minutes every day, prioritizing quality '
                'protein with your meals, and getting at least seven hours of restful sleep each night.'},
    {   'domain': 'ai_conversational_friend_recommendation',
        'label': 1,
        'text': 'If you are looking for an engaging science fiction book to read this weekend, I cannot recommend '
                'Project Hail Mary enough. The narrative pacing is fantastic, the scientific problem-solving feels '
                'grounded yet wonderfully imaginative, and the central relationship between the protagonist and an '
                'unexpected ally will genuinely warm your heart.'},
    {   'domain': 'ai_conversational_mindset_shift',
        'label': 1,
        'text': 'One of the most liberating realizations in adulthood is discovering that nobody really has everything '
                'figured out. Most people are simply navigating challenges one day at a time, making calculated '
                'decisions with incomplete information, and learning from their missteps along the way. Give yourself '
                'permission to be a work in progress.'},
    {   'domain': 'ai_conversational_closing_summary',
        'label': 1,
        'text': 'To wrap things up, developing strong communication skills is easily one of the highest-leverage '
                'investments you can make in your professional journey. The ability to articulate complex technical '
                'ideas clearly, listen with genuine empathy, and align cross-functional stakeholders will distinguish '
                'you far more than pure technical proficiency alone.'},
    {   'domain': 'quillbot_style',
        'label': 2,
        'text': "To be completely straightforward, I didn't anticipate the server starting up immediately on the "
                'initial attempt. I wasted three frustrating hours fixing an edge-case null pointer problem inside the '
                'connection pool handler.'},
    {   'domain': 'human_edited_ai',
        'label': 2,
        'text': "Artificial intelligence is definitely transforming modern enterprise workflows, but don't buy into "
                'all the marketing hype just yet. While deep neural networks uncover actionable insights from huge '
                'datasets, human oversight is still non-negotiable for reliable deployments.'},
    {   'domain': 'synonym_spun_essay',
        'label': 2,
        'text': 'In contemporary society, sustainable power sources have garnered significant consideration as viable '
                'choices over petroleum fuels. Photovoltaic solar mechanisms, explicitly, supply notable capacity for '
                'curbing greenhouse emissions.'},
    {   'domain': 'grammarly_polished_human',
        'label': 2,
        'text': 'We decided to revert the deployment within four minutes due to automated paging alerts. Pods '
                'throughout the cluster suffered severe CPU starvation because an inefficient regular expression '
                'caused catastrophic backtracking in our authorization routine.'},
    {   'domain': 'ai_draft_with_human_quips',
        'label': 2,
        'text': 'Diversification remains a cornerstone of sensible risk management in modern investment portfolios. '
                'You essentially avoid putting all your eggs in one basket, which protects your capital when market '
                'volatility inevitable spikes.'},
    {   'domain': 'hybrid_student_essay',
        'label': 2,
        'text': 'Renewable energy adoption has accelerated globally due to favorable climate legislation. I personally '
                'believe local communities should receive tax credits for installing residential solar panels, which '
                'would significantly decentralize energy grids.'},
    {   'domain': 'paraphrased_literature',
        'label': 2,
        'text': 'The tragic downfall of King Lear stems directly from his vain demand for hyperbolic flattery from his '
                'daughters. By banishing Cordelia, the sole honest child, he initiates his own descent into madness on '
                'the stormy heath.'},
    {   'domain': 'lightly_polished_technical',
        'label': 2,
        'text': 'The platform architecture leverages a microservice pattern to decouple deployment cycles across '
                'independent teams. Although operational overhead increases initially, organizational agility improves '
                'markedly as engineering teams scale.'},
    {   'domain': 'ai_rephrased_anecdote',
        'label': 2,
        'text': 'My grandmother frequently prepared sourdough loaves during cool weekend mornings. The entire '
                'residence was filled with the aroma of roasted crust and melted butter, a memory I cherish dearly '
                'even though I cannot recreate it.'},
    {   'domain': 'human_softened_ai_speech',
        'label': 2,
        'text': "Effective time management is essential for career longevity, but let's be real: strict productivity "
                "matrixes don't work for everyone. Finding a sustainable daily cadence matters far more than following "
                'rigid organizational formulas.'},
    {   'domain': 'paraphrased_health_advice',
        'label': 2,
        'text': 'Living a well-rounded healthy life demands proper dietary choices combined with consistent workout '
                'routines. Eating whole foods naturally boosts bodily immunity, and regular cardiovascular exercise is '
                'great for decompressing after stressful days.'},
    {   'domain': 'quillbot_customer_response',
        'label': 2,
        'text': 'We genuinely appreciate you reaching out regarding the delay with your parcel. We realize how '
                'critical prompt arrival is and deeply regret this setback. Our team is actively reviewing your '
                'shipment status right now.'},
    {   'domain': 'ai_with_slang_injections',
        'label': 2,
        'text': 'Digital learning tools have drastically changed higher education, though tbh, sitting through '
                'two-hour Zoom lectures is brutal. Still, having asynchronous access to course materials makes '
                'juggling part-time work much easier.'},
    {   'domain': 'paraphrased_environmental',
        'label': 2,
        'text': 'Global climate shifts pose severe threats to planetary ecosystems worldwide. Rising sea temperatures '
                'accelerate catastrophic storms, which is why immediate international decarbonization accords are so '
                'urgently required.'},
    {   'domain': 'edited_marketing_draft',
        'label': 2,
        'text': 'Experience truly remarkable sound fidelity with our latest wireless headphones. We engineered these '
                'with custom acoustic drivers to deliver punchy bass and clean highs without feeling heavy on your '
                'head after hours of use.'},
    {   'domain': 'paraphrased_philosophy',
        'label': 2,
        'text': 'The debate over free will continues to divide neuroscientists and ethicists. While hard determinists '
                'argue that physical brain chemistry governs every choice, compatibilists maintain that individuals '
                'still bear personal responsibility for deliberate actions.'},
    {   'domain': 'human_edited_cybersecurity',
        'label': 2,
        'text': 'Deploying multi-factor authentication is one of the easiest ways to safeguard company accounts. '
                'Requiring a secondary one-time code stops credential-stuffing attacks dead in their tracks even if an '
                'employee clicks a phishing link.'},
    {   'domain': 'paraphrased_travel_guide',
        'label': 2,
        'text': 'Kyoto stands out for its serene heritage, boasting historic wooden shrines and tranquil moss gardens. '
                'Travelers can take part in authentic tea rituals while wandering through peaceful bamboo paths away '
                'from the bustling city center.'},
    {   'domain': 'spun_academic_summary',
        'label': 2,
        'text': 'Past academic investigations highlight a meaningful relationship between early reading habits and '
                'child literacy development. Consistent exposure to rich literature fosters strong vocabulary '
                'acquisition and imaginative thinking.'},
    {   'domain': 'edited_smart_city',
        'label': 2,
        'text': 'Smart traffic control systems use connected sensor arrays to dynamically adjust intersection signal '
                'timings. This smooths out congestion during peak morning hours and noticeably cuts down on vehicle '
                'idling emissions across town.'},
    {   'domain': 'human_edited_nutrition',
        'label': 2,
        'text': 'Staying properly hydrated is fundamental for overall health. Drinking plenty of water supports kidney '
                'filtration and helps keep your energy steady, especially if you spend long hours working in '
                'air-conditioned offices.'},
    {   'domain': 'paraphrased_cloud_tech',
        'label': 2,
        'text': 'Cloud architectures give companies on-demand access to virtualized server hardware with pay-as-you-go '
                'pricing. This eliminates the headache of maintaining physical server racks and allows engineering '
                'teams to ship features much faster.'},
    {   'domain': 'hybrid_blog_post',
        'label': 2,
        'text': 'Mindfulness-based exercises have gained widespread traction for reducing daily work anxiety. I '
                'started taking five-minute breathing pauses between meetings, and it surprisingly helped blunt that '
                'mid-afternoon burnout feeling.'},
    {   'domain': 'edited_blockchain_explainer',
        'label': 2,
        'text': 'At its core, a blockchain is simply a shared, cryptographic ledger verified by decentralized peer '
                'nodes. By removing reliance on centralized authorities, it opens up fascinating possibilities for '
                'transparent digital auditing.'},
    {   'domain': 'paraphrased_sleep_research',
        'label': 2,
        'text': 'During REM sleep cycles, our brains actively organize daily memories and regulate emotional '
                'equilibrium. Chronic sleep deprivation severely impedes cognitive focus, which is why protecting '
                'consistent bedtime routines is crucial.'},
    {   'domain': 'lightly_spun_quantum',
        'label': 2,
        'text': 'Quantum computers rely on quantum mechanics principles like superposition and quantum entanglement. '
                'Instead of simple binary ones and zeros, quantum bits evaluate complex mathematical states '
                'simultaneously, paving the way for advanced cryptography.'},
    {   'domain': 'human_edited_agriculture',
        'label': 2,
        'text': 'Modern precision agriculture integrates soil sensors and drone mapping to pinpoint exactly where '
                'crops need water and nitrogen. This targeted approach saves farmers substantial input costs while '
                'preventing excess chemical runoff into local rivers.'},
    {   'domain': 'hybrid_conclusion',
        'label': 2,
        'text': 'To wrap things up, integrating artificial intelligence into everyday software offers immense promise, '
                "but we can't ignore the ethical risks. Balancing rapid innovation with prudent regulatory guidelines "
                'is the real challenge ahead.'},
    {   'domain': 'quillbot_intro_rewrite',
        'label': 2,
        'text': 'In our fast-moving technological landscape, adapting quickly to digital transformation is more '
                'critical than ever. Both professionals and businesses must commit to ongoing education if they wish '
                "to stay competitive in today's marketplace."},
    {   'domain': 'human_revised_ai_fiction',
        'label': 2,
        'text': 'The sleepy town of Oakridge was tucked away between gentle pine-covered hills and a winding mountain '
                'creek. Every dawn, the church steeple chimed six times, shaking the morning mist off the weathered '
                'cobblestone streets.'},
    {   'domain': 'paraphrased_casual',
        'label': 2,
        'text': 'yesterday my brother and i went over to the market to purchase apples to eat but heavy rain forced us '
                'to sprint back home'},
    {   'domain': 'paraphrased_tech_runon',
        'label': 2,
        'text': 'automated machine intelligence improves company operations by streamlining difficult data processing '
                'pipelines across virtual network clouds'},
    {   'domain': 'spun_casual_thought',
        'label': 2,
        'text': 'i happened to be strolling along the pathway when i noticed a canine barking loudly which caused me '
                'to switch sides'},
    {   'domain': 'lightly_edited_slang',
        'label': 2,
        'text': 'honestly man that film finale was pretty terrible to be honest who even penned that script'},
    {   'domain': 'quillbot_corporate',
        'label': 2,
        'text': 'In our fast-paced corporate world, getting different teams to collaborate smoothly is crucial for '
                'staying nimble. When companies adopt integrated project tools, distributed teams can actually '
                "coordinate without stepping on each other's toes. Plus, clear communication guidelines cut down on so "
                'much wasted effort.'},
    {   'domain': 'human_edited_ml',
        'label': 2,
        'text': 'Basically, machine learning breaks down into three core camps: supervised, unsupervised, and '
                'reinforcement learning. Supervised models rely on labeled examples to make predictions, while '
                'unsupervised setups hunt for hidden patterns without any human guidance.'},
    {   'domain': 'student_edited_ecology',
        'label': 2,
        'text': "Protecting biodiversity is honestly critical for keeping our planet's ecosystems in balance. When an "
                "ecosystem is diverse, it bounces back much faster from droughts and heatwaves. That's why global "
                'climate pacts are finally paying attention to habitat destruction.'},
    {   'domain': 'paraphrased_seo',
        'label': 2,
        'text': 'To rank higher on Google, you really need to create content that answers what people are actually '
                'searching for. Using clean heading tags and sharp meta descriptions makes it much easier for crawlers '
                "to index your pages. Better page speeds also mean visitors won't bounce right away."},
    {   'domain': 'lightly_spun_ev',
        'label': 2,
        'text': 'The shift toward electric cars is a massive milestone in cleaning up global emissions. Battery tech '
                'keeps getting better and cheaper, which is why almost every major automaker is pouring billions into '
                'electric platforms right now.'},
    {   'domain': 'human_softened_therapy',
        'label': 2,
        'text': 'Cognitive behavioral therapy is essentially about spotting harmful thought loops and swapping them '
                'with healthier habits. Clinical trials show that over time, patients genuinely learn to manage '
                'anxiety better without feeling overwhelmed.'},
    {   'domain': 'paraphrased_microservices',
        'label': 2,
        'text': 'Microservice architectures break down bulky monoliths into modular services that teams can deploy '
                'separately. Because each service handles a specific business function, engineers can scale '
                'traffic-heavy features without risking the whole application.'},
    {   'domain': 'hybrid_nuclear_summary',
        'label': 2,
        'text': 'To sum things up, nuclear power is gaining renewed attention as a reliable source of clean baseload '
                'energy. High startup costs and bureaucratic red tape remain big obstacles, but small modular reactors '
                'might be our best bet for hitting net-zero targets.'},
    {   'domain': 'human_edited_finance',
        'label': 2,
        'text': 'Solid financial planning starts with the basics: building a realistic budget and stashing three '
                'months of expenses in an emergency fund. Tossing spare cash into index funds takes advantage of '
                'compound interest, which quietly builds wealth over the long haul.'},
    {   'domain': 'spun_vertical_farming',
        'label': 2,
        'text': 'Vertical farming takes crop cultivation into city skyscrapers using stacked indoor trays under LED '
                'lighting. It cuts water usage by ninety percent and eliminates long-distance trucking, which is huge '
                'for urban food independence.'},
    {   'domain': 'human_revised_ai_ethics',
        'label': 2,
        'text': "You can't talk about AI development without addressing ethical guardrails. When automated algorithms "
                'make high-stakes decisions without fairness audits, they inevitably mirror human prejudice. Tech '
                'companies need independent oversight, not just self-regulation.'},
    {   'domain': 'paraphrased_nlp_progress',
        'label': 2,
        'text': 'Modern language models have made astounding leaps across reading comprehension and translation tests. '
                'By processing huge text datasets through attention layers, these models pick up on subtle linguistic '
                'nuances that previous systems completely missed.'},
    {   'domain': 'human_edited_logistics',
        'label': 2,
        'text': "Building bulletproof supply chains means you can't rely on a single vendor in one country. Companies "
                'are now spreading production across multiple regions and using GPS tracking to catch shipping '
                'bottlenecks before store shelves go empty.'},
    {   'domain': 'paraphrased_telehealth',
        'label': 2,
        'text': 'Telemedicine has completely changed the game for rural healthcare. Patients who used to drive four '
                'hours for a specialist appointment can now hop on a secure video call from their living room, which '
                'saves time and catches illnesses much earlier.'},
    {   'domain': 'lightly_spun_mindfulness',
        'label': 2,
        'text': 'Taking ten minutes for mindfulness meditation actually helps dial down workplace stress. Breathing '
                'calmly for a few moments triggers your parasympathetic system, which helps you stay level-headed when '
                'project deadlines get hectic.'},
    {   'domain': 'human_edited_ar',
        'label': 2,
        'text': 'Augmented reality blends physical spaces with digital overlays right through your phone or smart '
                "glasses. Whether it's showing factory technicians how to repair a turbine or giving medical students "
                '3D anatomy models, spatial computing is definitely here to stay.'},
    {   'domain': 'paraphrased_chat_advice',
        'label': 2,
        'text': 'You know, sometimes life puts you in this strange phase where you want to achieve big things but '
                "don't quite know the exact path yet. You're trying new skills and studying hard, but wonder if any of "
                "it is actually paying off. Honestly, don't stress about having every detail figured out at twenty. "
                'Just focus on becoming someone your future self respects.'},
    {   'domain': 'paraphrased_api_waiter',
        'label': 2,
        'text': 'A great way to understand APIs is to picture a restaurant waiter. You sit down and check the menu, '
                'while the kitchen cooks everything in the back. Rather than wandering into the kitchen yourself, you '
                "tell the waiter what you want, and they bring your plate back. That's pretty much how APIs handle web "
                'requests between clients and servers.'},
    {   'domain': 'paraphrased_exam_stress',
        'label': 2,
        'text': "If exam stress is getting to you today, just pause and take a breather. It's totally normal to feel "
                'swamped when deadlines pile up out of nowhere. Instead of trying to cram the whole book at once, '
                'break your study sessions into short twenty-minute bursts. Small steady steps will get you across the '
                'finish line.'},
    {   'domain': 'paraphrased_debugging_break',
        'label': 2,
        'text': 'Debugging can be crazy annoying, especially after staring at the same few lines for hours without '
                'finding the issue. Honestly, stepping away from the screen for five minutes does wonders. Grab some '
                'water, walk around the room, and usually the fix jumps out right when you sit back down.'},
    {   'domain': 'paraphrased_morning_routine',
        'label': 2,
        'text': "A solid morning routine isn't about waking up at dawn to run miles. It's really just about starting "
                'your day with a clear head instead of instantly grabbing your phone and doomscrolling social media. '
                'Even ten minutes sipping coffee and jotting down a quick to-do list sets a much better tone for the '
                'day.'},
    {   'domain': 'paraphrased_student_partnership',
        'label': 2,
        'text': 'Super burnt out with midterms coming up this week. We have two heavy exams on Thursday and a software '
                'engineering project due on Friday, but our teammate has gone completely MIA on WhatsApp. Really '
                'hoping we make it to semester break in one piece.'},
    {   'domain': 'paraphrased_productivity_reset',
        'label': 2,
        'text': "Being productive doesn't mean packing every minute of your day until you hit a wall. Real "
                'productivity is knowing when to work hard and when to step back and recharge. If simple tasks start '
                "dragging on forever, that's just your mind telling you it needs a real break."},
    {   'domain': 'paraphrased_career_shift',
        'label': 2,
        'text': "When you're stuck choosing between two career paths, don't get trapped over-analyzing every single "
                'detail. A good rule of thumb is picking the option that challenges you more and surrounds you with '
                'smarter colleagues. Good pay and promotions usually follow once you build real expertise.'},
    {   'domain': 'paraphrased_fitness_habit',
        'label': 2,
        'text': "Getting in shape doesn't require extreme workout routines or crazy diets. The routine that actually "
                'works is the one you can stick with for months without burning out. Just try walking daily, eating '
                'clean protein, and aiming for seven hours of good sleep.'},
    {   'domain': 'paraphrased_novel_review',
        'label': 2,
        'text': "Just wrapped up reading East of Eden and I'm still processing it. The villain's characterization is "
                'genuinely chilling, but the closing discussion on human choice gave me real goosebumps. Definitely '
                "one of the top books I've picked up recently."},
    {   'domain': 'paraphrased_learning_curve',
        'label': 2,
        'text': "Picking up a new programming framework is always frustrating in the beginning. You're going to hit "
                'weird syntax errors and struggle with buggy documentation. Just remember that every experienced '
                'developer started at zero too. The key is just having the patience to keep troubleshooting.'},
    {   'domain': 'paraphrased_apartment_hunt',
        'label': 2,
        'text': 'Renting an apartment in this neighborhood has become ridiculous. Checked out a tiny studio where the '
                'shower stall was crammed right next to the stove, and the agent still expected three months rent '
                'upfront with zero shame.'},
    {   'domain': 'paraphrased_keyboard_blunder',
        'label': 2,
        'text': 'Tip of the day: never try dusting your keyboard while your computer is unlocked. Ended up firing off '
                "random keyboard gibberish directly into the company general channel and had to explain it wasn't a "
                'cyberattack.'},
    {   'domain': 'paraphrased_city_pop',
        'label': 2,
        'text': 'Spotify shuffled into a retro Japanese city pop station while I was doing chores and I got hooked '
                'immediately. The bass grooves and nostalgic synth layers on these tracks are surprisingly catchy.'},
    {   'domain': 'paraphrased_communication_skills',
        'label': 2,
        'text': 'To sum it up, polishing your communication skills pays massive dividends across your career. Being '
                'able to explain technical concepts in plain English and work smoothly with non-engineers will take '
                'you way further than coding speed alone.'}]
