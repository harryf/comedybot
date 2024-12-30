# Calculating Laughs Per Minute

## General Approach to Calculating Laughs Per Minute

The following is a description of how to score a comedy set by analysing the audience reaction to the jokes.

- LOUD laughter and EVERYONE applauds = 5 points
- LOUD laughter and a SMATTERING of applause = 4 points
- LOUD laughter from EVERYONE and NO applause = 3 points
- MEDIUM laughs = 2 points
- SMATTERING of laughs = 1 point

To calculate the laughs per minute...
1. Add up your total laughter points.
2. How many minutes were you onstage?
3. Divide total laughter points by amount of time on stage to get LPM

To grade the score...
- 12 to 20 LPM: You are rockin', baby, and if you aren't making the big bucks yet-you will.
9 to 12 LPM: You're doing well and are ready to get paid, but think about shortening your setups.
6 to 9 LPM: Not bad, but not ready for the big time.
Below 6 LPM: Something ain't working.

## Calculating the Laughs Per Minute from the Audience Reactions JSON

JSON structure of reactions with a start and end time. Much of the time there will be no reaction, so the structure will have no "audio_tags". Sometimes though the crowd will be laughing or similar. Each element of the audience reactions has a "reaction_score" - a value from 0 (no reaction) to 5 (best reaction - an applause break). By looking at where the audience reactions are, you should also be able to determine how this correlates to the jokes. Typically the setup will not have a direct reaction.

The following as an example of part of the audience reactions JSON;

{
    "summary": {
        "total_score": 233,
        "laughs_per_minute": 18
    },
    "reactions": [
(snip)
        {
            "start": 50.4,
            "end": 51.6,
            "audio tags": [
                "Snicker",
                "Chuckle, chortle"
            ],
            "reaction_score": 2,
            "cumulative_score": 11
        },
        {
            "start": 51.6,
            "end": 52.8,
            "audio tags": [
                "Snicker",
                "Chuckle, chortle"
            ],
            "reaction_score": 2,
            "cumulative_score": 13
        },
(snip)

By detecting the start and end time of a bit from looking at the the structure of the material then double checking against the audience reactions around the same time, it should be possible to detect bits quite accurately.

Also given a start and end time, you can add up the "reaction_score" values then time by the difference between the start and end time of the bit, to calculate the laughs per minute (lpm) value of the bit.